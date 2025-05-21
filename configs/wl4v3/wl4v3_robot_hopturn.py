from isaacgym.torch_utils import *
import torch
from isaacgym import gymtorch, gymapi, gymutil
# config
from configs.base.legged_robot import LeggedRobot
def quat_to_rot_matrix(quat):
    """
    将四元数转换为旋转矩阵。

    Args:
        quat (torch.Tensor): 输入的四元数张量，形状为 (N, 4)，其中 N 是样本数量。

    Returns:
        torch.Tensor: 输出的旋转矩阵张量，形状为 (N, 3, 3)。
    """
    x, y, z, w = quat.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)
    ], dim=-1).reshape(-1, 3, 3)
class Wl4V3RobotHopturn(LeggedRobot):
    #------------ enviorment core ----------------
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
        self.start_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _post_physics_step_callback(self):
        self.dof_pos[:,self.foot_joint_indices]  = 0 
        super()._post_physics_step_callback()
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs

        # rot_mat = quat_to_rot_matrix(self.base_quat)
        # self.reset_buf |= rot_mat[:, 2, 2] < 0.9 # 0.95

        self.reset_buf |= self.time_out_buf

    def reindex(self,tensor):
        #sim2real purpose
        return tensor[:,[4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]]
    
    def reindex_feet(self,tensor):
        return tensor[:,[1,0,3,2]]
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction
        # print(actions[0,:])
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P_AND_V":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]
                - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        # print(torques[0,:])
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def reset_idx(self, env_ids):
        self.start_yaw[env_ids] = 0
        super().reset_idx(env_ids)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos# * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    #------------ reward functions----------------

    # def _reward_yaw_control(self):
        # start_time = 0.5
        # finish_time = 1.0
        # current_yaw = get_euler_xyz(self.base_quat)[2]
        # current_yaw = torch.where(current_yaw > torch.pi, current_yaw - 2 * torch.pi, current_yaw) 
        # self.start_yaw = torch.where(self.current_time < start_time, current_yaw, self.start_yaw)
        # phase = (self.current_time - start_time)/(self.max_episode_length_s - start_time) * (self.current_time > start_time)
        # traget_yaw = self.start_yaw + 1 * torch.pi * 0.5
        # yaw_error = torch.abs(current_yaw - traget_yaw)
        # reward = torch.exp(-torch.square(yaw_error)) * (self.current_time > start_time) * (self.current_time < finish_time)
        # # print(reward)

        # return reward
        # return 0
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    # def _reward_yaw_control(self):
    #     current_time = self.episode_length_buf * self.dt
    #     phase = (current_time - 0.5).clamp(min=0, max=0.25)
    #     current_yaw = get_euler_xyz(self.base_quat)[2]
    #     current_yaw = torch.where((current_yaw > torch.pi), current_yaw - 2*torch.pi, current_yaw)
    #     # print(current_yaw)
    #     self.start_yaw = torch.where((phase < self.dt), current_yaw, self.start_yaw)
    #     target_yaw = torch.where((torch.logical_or(phase < self.dt, phase > (0.25 - self.dt))), current_yaw, self.start_yaw + phase * 2.0 * torch.pi)
    #     yaw_err = torch.square(target_yaw - current_yaw)
    #     return yaw_err * torch.logical_and(current_time > 0.5, current_time < 0.75)
    
    # def _reward_base_height(self):
    #     current_time = self.episode_length_buf * self.dt
    #     # Penalize base height away from target
    #     base_height = self._get_base_heights()
    #     return torch.square(base_height - self.cfg.rewards.base_height_target) * torch.logical_or(current_time < 0.5, current_time > 1.0)
    
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.abs(self.base_ang_vel[:, :2]), dim=1)
    
    # def _reward_lin_vel_xy(self):
    #     return torch.sum(torch.abs(self.base_lin_vel[:, :2]), dim=1)
    
    # def _reward_ang_vel_z(self):
    #     current_time = self.episode_length_buf * self.dt
    #     ang_vel_z = self.base_ang_vel[:, 2]
    #     penalty = torch.abs(ang_vel_z) * torch.logical_or(current_time < 0.5, current_time > 1.0)
    #     ang_vel_z = self.base_ang_vel[:, 2].clamp(max=3.0, min=-3.0)
    #     reward = ang_vel_z * torch.logical_and(current_time > 0.5, current_time < 1.0)
    #     return penalty + reward
    
    # def _reward_lin_vel_z(self):
    #     current_time = self.episode_length_buf * self.dt
    #     lin_vel_z = self.base_lin_vel[:, 2]
    #     penalty = torch.abs(lin_vel_z) * torch.logical_or(current_time < 0.5, current_time > 1.0)
    #     lin_vel_z = self.base_lin_vel[:, 2].clamp(max=1.5)
    #     reward = lin_vel_z * torch.logical_and(current_time > 0.5, current_time < 0.75)
    #     return penalty + reward

    # def _reward_feet_all_contact(self):
    #     current_time = self.episode_length_buf * self.dt
    #     feet_air = self.contact_forces[:, self.feet_indices, 2] < 1.
    #     return torch.any(feet_air, dim=1) * torch.logical_or(current_time < 0.5, current_time > 1.0)
    
    # def _reward_feet_distance_y(self):
    #     current_time = self.episode_length_buf * self.dt
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])

    #     stance_width = 0.48 * torch.ones([self.num_envs, 1,], device=self.device)
    #     desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
    #     stance_diff = torch.square(desired_ys - footpos_in_body_frame[:, :, 1]).sum(dim=1)
        
    #     return stance_diff
        
    # def _reward_feet_distance_x(self):
    #     current_time = self.episode_length_buf * self.dt
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])

    #     stance_width = 0.4 * torch.ones([self.num_envs, 1,], device=self.device)
    #     desired_ys = torch.cat([stance_width / 2, stance_width / 2, -stance_width / 2, -stance_width / 2], dim=1)
    #     stance_diff = torch.square(desired_ys - footpos_in_body_frame[:, :, 0]).sum(dim=1)
    #     # print(stance_diff)
    #     return stance_diff
    # def _reward_survival(self):
    #     current_time = self.episode_length_buf * self.dt
    #     return current_time/self.max_episode_length_s


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        current_time = self.episode_length_buf * self.dt
        return torch.square(self.base_lin_vel[:, 2]) * torch.logical_or(current_time < 0.5, current_time > 0.95)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        current_time = self.episode_length_buf * self.dt
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target) * torch.logical_or(current_time < 0.5, current_time > 0.95)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        #return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_powers_dist(self):
        # Penalize power dist
        return torch.var(self.torques*self.dof_vel, dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        return  torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        current_time = self.episode_length_buf * self.dt 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) * torch.logical_or(current_time < 0.5, current_time > 0.95)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        #rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)
        #rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_vertical_contact(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2),dim=-1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    # def _reward_foot_clearance(self):
    #     foot_height = torch.mean(self.foot_positions[:, :, 2].unsqueeze(1).repeat(1,self.num_height_points,1) - self.measured_heights.unsqueeze(2), dim=1)
    #     foot_xy_vel = torch.norm(self.foot_velocities[:,:,:2],dim=-1)
    #     target_height = 0.1 + 0.02
    #     rew_foot_clearance = torch.sum(torch.square(target_height - foot_height) * foot_xy_vel,dim=-1)
    #     return rew_foot_clearance

     
    # def _reward_foot_clearance(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
    #     return torch.sum(height_error * foot_leteral_vel, dim=1)

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        #no_contact = 1.*(self.contact_filt == 0)

        clearance_reward = height_error * foot_leteral_vel 
        
        return torch.sum(clearance_reward, dim=1)
    
    def _reward_foot_slide(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        
        cost_slide = torch.sum(self.contact_filt * foot_leteral_vel, dim=1)
        return cost_slide
    
    def _reward_foot_clearance_hippos(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        hip_pos_scale = (1 + torch.abs(self.dof_pos[:, [0, 3, 6, 9]]))
        return torch.sum(hip_pos_scale * height_error * foot_leteral_vel, dim=1)
    
    def _reward_foot_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.exp(-1*(footpos_in_body_frame[:, :, 2] + self.cfg.rewards.base_height_target)/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1)
        no_contact = 1.*(self.contact_filt == 0)
        return torch.sum(torch.clamp(height_error,0,1) * no_contact, dim=1)
    
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # flag = 1.*(torch.abs(self.commands[:,1]) == 0)
        # return flag * torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
    
    def _reward_phase_contact(self):
        contact_goal = 1.*(torch.sin(self.phase) > 0.0)
        return torch.mean(torch.abs(1.*self.contact_filt - contact_goal),dim=1)
    
    def _reward_phase_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.mean(height_point_flag * height_error, dim=1)
    
    def _reward_foot_swing_clearance(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        no_contact = 1.*(self.contact_filt == 0)

        return torch.sum(height_error * no_contact, dim=1)
    
    
    # def _reward_foot_clearance(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)

    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
 
    #     foot_leteral_vel = foot_leteral_vel * (1 + contact_filt)

    #     return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_foot_width_equlity(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return 1.*(torch.abs(self.commands[:,1]) == 0)*torch.square(width_1 - width_2)
    
    def _reward_foot_dia_enforce(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        dia_1 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,0,:] - footpos_in_body_frame[:,2,:]),dim=-1))
        dia_2 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,1,:] - footpos_in_body_frame[:,3,:]),dim=-1))

        return (torch.square(dia_1 - 0.51) + torch.square(dia_2 - 0.51))/2
    
    def _reward_foot_width_cons(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return (torch.square(width_1 - 0.3) + torch.square(width_2 - 0.3))/2.
    
    
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        flag = 1.*(torch.abs(self.commands[:,1]) == 0)
        return flag * torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        #return flag * 1.*(torch.abs(torch.sum(self.dof_pos[:, [0, 3, 6, 9]],dim=-1)) > 0.0)

    def _reward_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[4,5,6]] - self.dof_pos[:,[8,9,10]]),dim=-1)
        # diff3 = torch.sum(torch.square(self.dof_vel[:,3] - self.dof_vel[:,15]),dim=-1)
        # diff4 = torch.sum(torch.square(self.dof_vel[:,7] - self.dof_vel[:,11]),dim=-1)
        return 0.5*(diff1 + diff2)
    
    def _reward_trot_contact(self):
        contact_filt = 1.*self.contact_filt
        pattern_match1 = torch.mean(torch.abs(contact_filt - self.trot_pattern1),dim=-1)
        pattern_match2 = torch.mean(torch.abs(contact_filt - self.trot_pattern2),dim=-1)
        pattern_match_flag = 1.*(pattern_match1*pattern_match2 > 0)
        return pattern_match_flag*(torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_yaw_control(self):
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.25)
        current_yaw = get_euler_xyz(self.base_quat)[2]
        current_yaw = torch.where((current_yaw > torch.pi), current_yaw - 2*torch.pi, current_yaw)
        # print(current_yaw)
        self.start_yaw = torch.where((phase < self.dt), current_yaw, self.start_yaw)
        target_yaw = torch.where((torch.logical_or(phase < self.dt, phase > (0.25 - self.dt))), current_yaw, self.start_yaw + phase * 2.0 * torch.pi)
        yaw_err = torch.square(target_yaw - current_yaw)
        return yaw_err * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_lin_vel_z_1(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.base_lin_vel[:, 2].clamp(max=1.5)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.625)
    
    def _reward_ang_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = self.base_ang_vel[:, 2].clamp(max=3.0, min=-3.0)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_feet_distance_y(self):
        current_time = self.episode_length_buf * self.dt
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])

        stance_width = 0.48 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footpos_in_body_frame[:, :, 1]).sum(dim=1)
        return stance_diff
        
    def _reward_feet_distance_x(self):
        current_time = self.episode_length_buf * self.dt
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])

        stance_width = 0.4 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, stance_width / 2, -stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footpos_in_body_frame[:, :, 0]).sum(dim=1)
        return stance_diff