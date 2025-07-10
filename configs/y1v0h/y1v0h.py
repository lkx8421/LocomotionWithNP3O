from isaacgym.torch_utils import *
import torch
# env related
from configs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch

class Y1v0h(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4]
        self.foot_joint_indices = [3, 7]

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
 
        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras

    

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
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        # torques = torques * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if "tracking_lin_vel" not in self.reward_scales:
            if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_x"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            if torch.mean(self.episode_sums["tracking_lin_vel_y"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_y"]:
                self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
                self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

        elif torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.cfg.rewards.tracking_sigma)

    def _cost_acc_smoothness(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        acc_limit = self.dof_vel_limits/(2.*self.dt)
        return torch.mean(torch.max(torch.zeros_like(acc),torch.abs(acc) - acc_limit),dim=1)

    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)

    def _cost_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)

    def _reward_hip_pos(self):
        reward = torch.sum(torch.square(self.dof_pos[:, [0, 4]] - torch.zeros_like(self.dof_pos[:, [0, 4]])), dim=1)
        return reward

    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        reward = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1) 
        return reward 

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        two_or_single_contact = torch.sum(1.*contacts, dim=1)<2
        return 1.*two_or_single_contact
