from isaacgym.torch_utils import *
import torch
# env related
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

class ClimbRobot( LeggedRobot ):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
    def compute_observations(self):
        self.dof_pos[:, self.foot_joint_indices] = 0
        super().compute_observations()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_climb_condition()

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
        actions_scaled = actions[:, :16] * self.cfg.control.action_scale
        actions_scaled[:, [0, 4, 8, 12]] *= self.cfg.control.hip_scale_reduction
        # actions_scaled[:, [3, 7, 11, 15]] *= 20.0

        # if self.cfg.domain_rand.randomize_lag_timesteps:
        #     self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        #     joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        # else:
        #     joint_pos_target = actions_scaled + self.default_dof_pos

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        # joint_pos_target = torch.clamp(joint_pos_target,self.dof_pos-1,self.dof_pos+1)

        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[:,self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]
                - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[:,self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_climb_condition(self):
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        quat_only_yaw = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)

        cur_footvel_translated = self.feet_vel # feet velocity in move base frame ,but z axis always vertical of ground
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(quat_only_yaw, cur_footvel_translated[:, i, :])
        bool_front = torch.logical_and(self.commands[:, 0] > 0.1,  torch.any(footvel_in_body_frame[:, 0:2, 0] < 0.1, dim=1))
        bool_rear =  torch.logical_and(self.commands[:, 0] > 0.1,  torch.any(footvel_in_body_frame[:, 2:4, 0] < 0.1, dim=1))
        # position condition
        feet_pos_z = self.feet_pos[:, :, 2] - 0.0875 
        self.front_climb = torch.logical_and(bool_front, torch.any(feet_pos_z[:, 0:2] < -0.01, dim=1))
        self.rear_climb = torch.logical_and(bool_rear, torch.any(feet_pos_z[:, 2:4] < -0.01, dim=1))
        self.rear_climb *= ~self.front_climb
        # feet_contact_x = self.contact_forces[:, self.feet_indices, 0] < -1
        # bool_front = torch.any(feet_contact_x[:, 0:2], dim=1)
        # bool_rear = torch.any(feet_contact_x[:, 2:4], dim=1)
        # feet air
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        base_x_axis = torch.stack([
            1 - 2*self.base_quat[:, 1]**2 - 2*self.base_quat[:, 2]**2, 
            2*self.base_quat[:, 0]*self.base_quat[:, 1] + 2*self.base_quat[:, 3]*self.base_quat[:, 2], 
            2*self.base_quat[:, 0]*self.base_quat[:, 2] - 2*self.base_quat[:, 3]*self.base_quat[:, 1]
        ], dim=1).to(self.device)
        dot_product = torch.clip(torch.sum(base_x_axis * torch.tensor([0, 0, 1], device=self.device), dim=-1), -1, 1)
        angle_error = torch.acos(dot_product)
        
        extra_height = torch.where(angle_error > 0.45*torch.pi, torch.zeros_like(angle_error), 0.3 * torch.cos(angle_error))
        base_height = base_height - extra_height
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_base_collision(self):
        # Penalize collisions on selected bodies
        return torch.norm(self.contact_forces[:, 0, 0:2], dim=-1) > 0.1
    
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
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
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
        rew_airTime *= torch.norm(self.commands[:, 1:3], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return 0.25 * torch.sum(contact, dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)/5)
        return reward * torch.all(contact, dim=1)
    
    def _reward_heading(self):
        if self.cfg.commands.heading_command:
            _, _, heading = get_euler_xyz(self.base_quat)
            heading = torch.where(heading > torch.pi, heading - 2 * torch.pi, heading) # limit heading to [-pi, pi]
            reward = torch.square(heading - self.commands[:, 3])
            return reward
        else:
            return 0
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    ########### add new below #############
    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)
    
    def _reward_climb_feet_air(self):
        feet_contact_z = self.contact_forces[:, self.feet_indices, 2] > 1.
        reward_front_air = -0.5*torch.sum(1.0*feet_contact_z[:, 0:2], dim=1)
        reward_rear_air = -0.5*torch.sum(1.0*feet_contact_z[:, 2:4], dim=1)
        reward_front = torch.where(self.front_climb, reward_front_air, torch.zeros_like(reward_front_air))
        reward_rear = torch.where(self.rear_climb, reward_rear_air, torch.zeros_like(reward_rear_air))
        return reward_front + reward_rear
    
    def _reward_climb_pitch(self):
        rot_mat = quat_to_rot_matrix(self.base_quat)
        base_x_world_z_angle = torch.acos(torch.clip(rot_mat[:, 2, 0], -1, 1))
        base_z_world_z_angle = torch.acos(torch.clip(rot_mat[:, 2, 2], -1, 1))
        reward_front_pitch = -torch.square(base_x_world_z_angle) # 0.25 * torch.exp(-torch.square(base_x_world_z_angle)/0.5) 
        reward_rear_pitch = -torch.square(base_z_world_z_angle) # 0.25 * torch.exp(-torch.square(base_z_world_z_angle)/0.5) 
        reward_front = torch.where(self.front_climb, reward_front_pitch, torch.zeros_like(reward_front_pitch))
        reward_rear = torch.where(self.rear_climb, reward_rear_pitch, torch.zeros_like(reward_rear_pitch))
        return reward_front + reward_rear

    def _reward_climb_feet_lift(self):
        # feet lift
        hip_pos = self.rigid_body_states[:, [1, 5, 9, 13], 0:3]
        cur_footpos_translated = self.feet_pos - hip_pos
        footpos_in_hip_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_hip_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        clearance_height_target = 0.1
        height_err = torch.norm(footpos_in_hip_frame, dim = 2) - clearance_height_target

        reward_front_height = torch.exp(-torch.mean(torch.square(height_err[:, 0:2]), dim = 1)/0.1)
        reward_rear_height = torch.exp(-torch.mean(torch.square(height_err[:, 2:4]), dim = 1)/0.1)

        reward_front = self.front_climb * reward_front_height
        reward_rear = self.rear_climb * reward_rear_height
        return reward_front + reward_rear
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05) +\
            torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05)
        return reward 

    def _reward_hip_pos(self):
        # penalty hip joint position not equal to zero
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, [0, 4, 8, 12]] - torch.zeros_like(self.dof_pos[:, [0, 4, 8, 12]])), dim=1)/0.05) 
        return reward # torch.sum(torch.square(self.dof_pos[:, [0, 4, 8, 12]] - torch.zeros_like(self.dof_pos[:, [0, 4, 8, 12]])), dim=1)
    
    def _reward_com_feet_contact(self):
        com = self.root_states[:, 0:2]
        feetpos_contact = torch.zeros(self.num_envs, 2, device=self.device)
        feetpos_all = torch.zeros(self.num_envs, 2, device=self.device)
        contact_num = torch.zeros(self.num_envs, device=self.device)
        for i in range(len(self.feet_indices)):
            feetpos_all += self.feet_pos[:, i, 0:2] 
            feetpos_contact += self.feet_pos[:, i, 0:2] * self.contact_filt[:, i].unsqueeze(1) 
            contact_num += self.contact_filt[:, i]
        feetcom = torch.where(contact_num.unsqueeze(1) > 0, feetpos_contact/contact_num.unsqueeze(1), feetpos_all/len(self.feet_indices))
        error = torch.sum(torch.square(com - feetcom), dim=1)

        return torch.exp(-error/0.25)        
    
    def _reward_feet_relative_x(self):
        hip_pos = self.rigid_body_states[:, [1, 5, 9, 13], 0:3]
        cur_footpos_translated = self.feet_pos - hip_pos
        footpos_in_hip_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_hip_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        target_foot_relative_x = torch.tensor([0.2, 0.2, 0.0, -0.0], device=self.device)
        rew_foot_relative_x = torch.exp(-torch.sum(torch.square(footpos_in_hip_frame[:, :, 0] - target_foot_relative_x), dim=1)/0.025)
        feet_contact_z = self.contact_forces[:, self.feet_indices, 2] > 1.

        return torch.where(torch.all(feet_contact_z, dim=1), rew_foot_relative_x, torch.zeros_like(rew_foot_relative_x))
        # return rew_foot_relative_x
        # return torch.where(self.episode_length_buf > self.threshold_episode_length, torch.zeros_like(rew_foot_relative_x), rew_foot_relative_x)

        # return torch.where(self.episode_length_buf > self.threshold_episode_length, result, 0.5*(front_penalty + rear_penalty))
    def _reward_feet_upper_height(self):
        feet_contact_z = self.contact_forces[:, self.feet_indices, 2] > 1.
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        reward_front = torch.exp(-torch.square(torch.mean(footpos_in_body_frame[:, 0:2, 2], dim=1) - -0.1)/0.05)
        reward_rear = 2.0*torch.exp(-torch.square(torch.mean(footpos_in_body_frame[:, 2:4, 2], dim=1) - -0.1)/0.05)

        return ~self.delta_flat*reward_front*torch.all(~feet_contact_z[:, 0:2], dim=1) + ~self.delta_flat*reward_rear*torch.all(~feet_contact_z[:, 2:4], dim=1)
    
    def _reward_feet_lin_pos_z(self):
        feet_pos_z = self.feet_pos[:, :, 2] - 0.0875
        feet_contact_z = self.contact_forces[:, self.feet_indices, 2] > 1.
        reward = torch.all(torch.logical_and(feet_pos_z[:,0:2] > 0, feet_contact_z[:,0:2]), dim=1) * 1.0 \
            + torch.all(torch.logical_and(feet_pos_z[:,2:4] > 0, feet_contact_z[:,2:4]), dim=1) * 4.0
        return reward
    
    def _reward_vertical_contact(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2),dim=-1)

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        clearance_height_target = -0.1
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        #no_contact = 1.*(self.contact_filt == 0)
    
        clearance_reward = height_error * foot_leteral_vel
    
        return torch.sum(clearance_reward, dim=1)