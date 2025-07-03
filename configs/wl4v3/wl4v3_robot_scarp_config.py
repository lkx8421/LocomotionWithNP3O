# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from isaacgym.torch_utils import *
import torch
# config
from configs.wl4v2.wl4v2_robot import *
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
class Wl4V3RobotScarp(Wl4V2Robot):
    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.reset_buf |= self._get_base_heights() < -0.5
        # distance = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1)
        # finished = distance > self.terrain.env_length / 2
        # self.reset_buf |= finished
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # print(self.torques[0,:])
    # rewards
    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.all(contact, dim=1)
    
    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 4])
        actions_diff += torch.square(self.actions[:, 1:2] - self.actions[:, 5:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 8] + self.actions[:, 12])
        actions_diff += torch.square(self.actions[:, 9:10] - self.actions[:, 13:14]).sum(dim=-1)
        return actions_diff 

    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05) +\
            torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05)
        penalty = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1) +\
            torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)
        return penalty
     
    def _reward_lin_vel_xy(self):
        current_time = self.episode_length_buf * self.dt
        error = (torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=-1) > 0.5) * 1.0 * (current_time > 0.2)
        return error
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)   
        # return lin_vel_error
    
    def _reward_base_lin_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_root_vel[:, :3] - self.root_states[:, 7:10]) / self.dt), dim=1)
    
    def _reward_heading(self):
        if self.cfg.commands.heading_command:
            _, _, heading = get_euler_xyz(self.base_quat)
            heading = torch.where(heading > torch.pi, heading - 2 * torch.pi, heading) # limit heading to [-pi, pi]
            reward = torch.square(heading - self.commands[:, 3])
            return reward
        else:
            return 0
        
    def _reward_hip_pos(self):
        # return torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]), dim=1)
        # return torch.exp(-torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - torch.zeros_like(self.dof_pos[:, self.hip_joint_indices])), dim=1)/0.05) 
        # flag = 1.#*(torch.abs(self.commands[:,1]) == 0)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - torch.zeros_like(self.dof_pos[:, self.hip_joint_indices])), dim=1)
        #return flag * 1.*(torch.abs(torch.sum(self.dof_pos[:, [0, 3, 6, 9]],dim=-1)) > 0.0)
    
    def _reward_feet_distance(self):
        cur_footsteps_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat,
                                                                 cur_footsteps_translated[:, i, :])

        stance_length = 0.4 * torch.ones([self.num_envs, 1,], device=self.device)
        stance_width = 0.5 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_xs = torch.cat([stance_length / 2, stance_length / 2, -stance_length / 2, -stance_length / 2], dim=1)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0]).sum(dim=1)
        stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        # return stance_diff_x + stance_diff_y
        return torch.exp((-stance_diff_x - stance_diff_y)/0.05)
    
    def _reward_feet_max_distance(self):
        cur_footsteps_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat,
                                                                 cur_footsteps_translated[:, i, :])

        place_x = torch.tensor([0.175, 0.175, -0.175, -0.175], device=self.device).repeat(self.num_envs, 1)  # 形状 [num_envs, 4]
        place_y = torch.tensor([0.275, -0.275, 0.275, -0.275], device=self.device).repeat(self.num_envs, 1) 
        place = torch.stack([place_x, place_y], dim=2)  # 合并为 [num_envs, 4, 2]
        distance = torch.norm(place - footsteps_in_body_frame[:, :, :2], dim=2)
        reward = torch.sum(1.0 * (distance > 0.05), dim=1) / 4
        # print(distance)
        return reward
    def _reward_foot_clearance(self):
        current_time = self.episode_length_buf * self.dt

        target_height = -0.1
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        stance_width = 0.5 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        exf = torch.abs(footpos_in_body_frame[:, :, 1] - desired_ys) > 0.025
        # print(torch.abs(footpos_in_body_frame[:, :, 1] - desired_ys))
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)

        no_contact = self.contact_forces[:, self.feet_indices, 2] < 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        # no_contact = 1.*(contact == 0)

        reward = torch.exp(-height_error * foot_leteral_vel/0.1) * no_contact * exf
        reward = torch.sum(reward, dim=1) / len(self.feet_indices) * (current_time > 0.4)
        # clearance_reward = torch.sum( , dim=1) 
        # print((height_error * foot_leteral_vel)[0,:])
        # clearance_reward = torch.exp(-clearance_reward/0.02)
        return reward
    
    def _reward_finished(self):
        distance = torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1)
        max_distance = self.terrain.env_length / 2
        # print(distance)
        # distance
        # print(torch.clip(distance - 2.0, min=0., max=max_distance - 2.0))
        return torch.clip(distance - 2.0, min=0., max=max_distance - 2.0) * self.terrain_levels/10
    
class Wl4V3RobotScarpCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 16 + 16 + 16
        n_proprio = 60 #
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 16
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.60] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.,
                     'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 1.0
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True 
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-2.0, 2.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/wl4v3/urdf/robot.urdf'
        foot_name = "foot"
        name = "wl4v3"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            # lin_vel_xy = -6.0
            orientation = -0.2
            ang_vel_xy = -0.05
            dof_vel = 0.0
            dof_acc = -2.5e-7
            # base_lin_acc = -0.01
            base_height = -1.0
            feet_air_time = 0.
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            action_smoothness= 0.0#-0.002
            foot_mirror = -0.5
            # stand_still = -1.0
            # heading = -0.05

            hip_pos = -0.5
            # feet_contact_forces = -1.0
            # torque_limits = -1.0
            # actions_symmetry = -0.01
            # feet_distance = 0.7
            # feet_max_distance = -0.5
            # foot_clearance = 1.0
            finished = 1.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        max_contact_force = 250.  # forces above this value are penalized

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = True
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        # randomize_initial_joint_pos = True
        # initial_joint_pos_range = [0.5, 1.5]
    
    class costs:
        num_costs = 3
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        slope_treshold = 2.0  # slopes above this threshold will be corrected to vertical surfaces

        amplitude = [0.4, 1.5]
        # step_height = [0.05, 0.23]
        # discrete_obstacles_height = [0.05, 0.25]
        
class Wl4V3RobotScarpCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 3

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'wl4v3_scarp'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 5000
        num_steps_per_env = 24
        resume = False
        resume_path = ''
 

  
