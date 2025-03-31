from isaacgym.torch_utils import *
import torch
# env related
from configs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch

# def random_quat(U):
#     u1 = U[:,0].unsqueeze(1)
#     u2 = U[:,1].unsqueeze(1)
#     u3 = U[:,2].unsqueeze(1)
#     q1 = torch.sqrt(1-u1)*torch.sin(2*torch.pi*u2)
#     q2 = torch.sqrt(1-u1)*torch.cos(2*torch.pi*u2)
#     q3 = torch.sqrt(u1)*torch.sin(2*torch.pi*u3)
#     q4 = torch.sqrt(u1)*torch.cos(2*torch.pi*u3)
#     Q = torch.cat([q1,q2,q3,q4],dim=-1)
#     return Q

class TitaRobot(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4]
        self.foot_joint_indices = [3, 7]

    def reindex(self,tensor):
        #sim2real purpose
        return tensor
    
    def reindex_feet(self,tensor):
        return tensor

    def _post_physics_step_callback(self):
        self.dof_pos[:, self.foot_joint_indices] = 0
        super()._post_physics_step_callback()

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
        # actions_scaled[:, self.foot_joint_indices] *= self.cfg.control.foot_scale_reduction

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
        # torques = torques * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # def _reset_root_states(self, env_ids):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         Sets base position based on the curriculum
    #         Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """
    #     # base position
    #     if self.custom_origins:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #         self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    #     else:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #     # base velocities
    #     # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
    #     # # random ori
    #     # self.root_states[env_ids, 3:7] = random_quat(torch_rand_float(0, 1, (len(env_ids), 4), device=self.device))
    #     # random height
    #     # self.root_states[env_ids, 2:3] += torch_rand_float(0, 0.2, (len(env_ids), 1), device=self.device)
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    

    #------------ reward functions----------------
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        reward_pos = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        reward_vel = 0.1*torch.sum(torch.abs(self.dof_vel), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return reward_pos# + reward_vel
    
    def _reward_vertical_contact(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2),dim=-1)    
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

    def _reward_com_feet(self):
        com = self.root_states[:, 0:2]
        feet_com = (self.feet_pos[:, 0, 0:2] + self.feet_pos[:, 1, 0:2])/2
        err = torch.sum(torch.square(com - feet_com), dim=1)
        return torch.exp(-err/0.25)
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05) 
        return reward 

    def _reward_hip_pos(self):
        # penalty hip joint position not equal to zero
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, [0, 4]] - torch.zeros_like(self.dof_pos[:, [0, 4]])), dim=1)/0.05) 
        return reward
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact
    
    def _reward_feet_distance(self):
        feet_state = self.rigid_body_states[:, self.feet_indices, :]
        feet_distance = torch.abs(torch.norm(feet_state[:, 0, :2] - feet_state[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        # return (~self.reset_buf).float() * self.dt
        return (self.episode_length_buf * self.dt) > 10
    
    def _reward_leg_symmetry(self):
        foot_positions_base = self.feet_pos - \
                            (self.root_states[:, 0:3]).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ 0.001)
    def _reward_wheel_adjustment(self):
        # 鼓励使用轮子的滑动克服前后的倾斜，奖励轮速和倾斜方向一致的情况，并要求轮速方向也一致
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.feet_vel[:, 0, 0] + self.feet_vel[:, 1, 0]) / 2
        # 两边轮速不一致的情况，不给奖励
        wheel_x_invalid = (self.feet_vel[:, 0, 0] * self.feet_vel[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)
        reward = incline_x * wheel_x_mean > 0
        return reward
    #------------ cost functions----------------
    """
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

    """
    def _cost_torque_limit(self):
        # constaint torque over limit
        #return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
        # return 1.*(torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)>0.0)
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _cost_pos_limit(self):
        # upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        # lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        # out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        # return out_limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # return 1.*(torch.sum(out_of_limits, dim=1)>0.0)
        return torch.sum(out_of_limits, dim=1)
   
    def _cost_dof_vel_limits(self):
        # return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
        # return 1.*(torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)>0.0)
         return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _cost_vel_smoothness(self):
        return torch.mean(torch.max(torch.zeros_like(self.dof_vel),torch.abs(self.dof_vel) - (self.dof_vel_limits/2.)),dim=1)
    
    def _cost_acc_smoothness(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        acc_limit = self.dof_vel_limits/(2.*self.dt)
        return 0.1*torch.mean(torch.max(torch.zeros_like(acc),torch.abs(acc) - acc_limit),dim=1)
    
    def _cost_collision(self):
        return  torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)
        # return torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
    
    def _cost_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)

    def _cost_base_height(self):
        # Penalize base height away from target
        # base_height = self._get_base_heights()
        # return 1.*(torch.abs(base_height) < self.cfg.rewards.base_height_target) #+ 1.*(torch.abs(base_height) > self.cfg.rewards.base_height_target) 
        # base_height = self._get_base_heights()
        # return torch.square(base_height - self.cfg.rewards.base_height_target)
        base_height = self._get_base_heights()
        # return 1.*(torch.square(base_height - self.cfg.rewards.base_height_target) > 0.0) 
        return 100*torch.square(base_height - self.cfg.rewards.base_height_target)
    
    
    def _cost_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
       
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return torch.max(torch.zeros_like(rew_airTime),-1.*rew_airTime)#1.*(rew_airTime < 0.0)
    
    def _cost_ang_vel_xy(self):
        ang_vel_xy = 0.01*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return ang_vel_xy
    
    def _cost_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _cost_torques(self):
        # Penalize torques
        torque_squres = 0.0001*torch.sum(torch.square(self.torques),dim=1)
        return torque_squres
    
    def _cost_action_rate(self):
        action_rate = 0.01*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate
    
    def _cost_walking_style(self):
        # number of contact must greater than 2 at each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        return 1.*(torch.sum(1.*contact_filt,dim=-1) < 3.)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_start_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _cost_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # return flag * torch.mean(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - 0.0),dim=-1)
    
    def _cost_feet_height(self):
        # Reward high steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        foot_heights_cost = torch.sum(torch.square(self.dof_pos[:,[2,5,8,11]] - (-2.0)) * (~contact_filt),dim=1)
 
        return foot_heights_cost
    
    def _cost_contact_force_xy(self):
        contact_xy_force_norm = torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :2],dim=-1),dim=-1)
        return contact_xy_force_norm

    def _cost_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _cost_default_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _cost_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_slip
    
    def _cost_feet_contact_velocity(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        foot_velocities = torch.square(self.foot_velocities[:, :, 2].view(self.num_envs, -1))
        rew_contact_force = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_contact_force
    
    def _cost_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_foot_swing_clearance(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error *= ~self.contact_filt

        return 10*torch.sum(height_error, dim=1)
    
    def _cost_foot_swing_clearance_cum(self):
        # treat foot as swing when no contact
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        no_contact = 1.*(1.*self.contact_filt == 0)

        return torch.mean(torch.abs(footpos_in_body_frame[:, :, 2]) * no_contact, dim=1)
    
    def _cost_foot_slide(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        
        cost_slide = torch.mean(self.contact_filt * foot_leteral_vel, dim=1)
        return cost_slide
    
    def _cost_trot_contact(self):
        contact_filt = 1.*self.contact_filt
        pattern_match1 = torch.mean(torch.abs(contact_filt - self.trot_pattern1),dim=-1)
        pattern_match2 = torch.mean(torch.abs(contact_filt - self.trot_pattern2),dim=-1)
        pattern_match_flag = 1.*(pattern_match1*pattern_match2 > 0)
        return pattern_match_flag*(torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    def _cost_phase_contact(self):
        contact_goal = 1.*(torch.sin(self.phase) > 0.0)
        return 1.*(torch.mean(torch.abs(1.*self.contact_filt - contact_goal),dim=1) > 0.0)
    
    def _cost_phase_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.sum(height_point_flag* height_error, dim=1)
    
    def _cost_phase_foot_min_height(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        heights = -1*footpos_in_body_frame[:, :, 2]
        height_point_flag = 1.*(torch.sin(self.phase) < 0.0)

        return torch.mean(height_point_flag* heights, dim=1)
    
    def _cost_foot_width(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        less_width = (1.*(width_1 < 0.28) + 1.*(width_2 < 0.28))/2
        greater_width = (1.*(width_1 > 0.31) + 1.*(width_2 < 0.31))/2

        return (less_width + greater_width)/2

    def _cost_foot_width_equlity(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        width_1 = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1])
        width_2 = torch.abs(footpos_in_body_frame[:,2,1] - footpos_in_body_frame[:,3,1])

        return torch.square(width_1 - width_2)

    def _cost_powers_dist(self):
        # Penalize power dist
        return 10e-5*torch.var(self.torques*self.dof_vel, dim=1)
    
    def _cost_idol_contact(self):
        contact_filt = 1.*self.contact_filt
        sum_contact_filt_flag = 1.*(torch.sum(contact_filt,dim=-1) < 4)
        idol_flag = 1.*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return idol_flag*sum_contact_filt_flag
    
    def _cost_idol_hip(self):
        idol_flag = 1.*(torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return idol_flag*torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - 0.0),dim=-1)
    
    def _cost_foot_dia_enforce(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        dia_1 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,0,:] - footpos_in_body_frame[:,2,:]),dim=-1))
        dia_2 = torch.sqrt(torch.sum(torch.square(footpos_in_body_frame[:,1,:] - footpos_in_body_frame[:,3,:]),dim=-1))

        return (torch.square(dia_1 - 0.51) + torch.square(dia_2 - 0.51))/2
    
    def _cost_foot_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.clamp(torch.exp(footpos_in_body_frame[:, :, 2]/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1),0,1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_foot_nocontact_regular(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
      
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.clamp(torch.exp(footpos_in_body_frame[:, :, 2]/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1),0,1)
        height_error *= ~self.contact_filt
        return torch.mean(height_error, dim=1)
    
    def _cost_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[9,10,11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[3,4,5]] - self.dof_pos[:,[6,7,8]]),dim=-1)
        return 0.05*(diff1 + diff2)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)




    
    
    
    

    
