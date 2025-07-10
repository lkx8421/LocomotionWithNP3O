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

from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.y1v0h.y1v0h import *

class Y1v0hCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        
        num_actions = 8
        n_scan = 187
        n_priv_latent =   2 + 1 + 4 + 1 + 1 + 8 + 8 + 8
        n_proprio = 36
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        # lin_vel = [0.0, 0.0, 0.0]  # x, y, z [m/s]
        # ang_vel = [0.0, 0.0, 0.0]  # x, y, z [rad/s]  
        default_joint_angles = {
                'FL_hip_joint': 0,
                'FR_hip_joint': 0,

                'FL_thigh_joint': 0.8,
                'FR_thigh_joint': 0.8,

                'FL_calf_joint': -1.5,
                'FR_calf_joint': -1.5,

                'FL_foot_joint': 0,
                'FR_foot_joint': 0,
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 40.,
                     'thigh_joint': 40.,
                     'calf_joint': 40.,
                     'foot_joint': 10.,
        }  # [N*m/rad]
        damping = {'hip_joint': 1.0,
                   'thigh_joint': 1.0,
                   'calf_joint': 1.0,
                   'foot_joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):

        file = '{ROOT_DIR}/resources/y1v0h/urdf/y1v0h_description.urdf'
        foot_name = "foot"
        name = "y1v0h"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
    
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 2.0
            tracking_lin_vel_x = 0.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation=-5.0
            # powers = -2e-5
            torques = -1e-05
            dof_pos_limits = -10.0
            torque_limits = 0.0
            dof_vel = 0.0
            dof_acc = -5e-7
            base_height = -10.0
            feet_air_time = 0.0
            collision = -10.0
            action_rate = -0.01
            # action_smoothness= 0
            stand_still = -1.0
            # no_fly = 0.5

            hip_pos = -2.0
            foot_mirror = -0.5

        soft_dof_pos_limit = 0.9
        base_height_target = 0.5

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.0]
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

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
    
    class costs:
        num_costs = 3
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1
            # vel_smoothness = 0.1
            # acc_smoothness = 0.1
            #collision = 0.1
            # feet_contact_forces = 0.1
            # stumble = 0.1
        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # vel_smoothness = 0.0
            # acc_smoothness = 0.0
            #collision = 0.0
            # feet_contact_forces = 0.0
            # stumble = 0.0        
    
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'plane'
        measure_heights = True
        include_act_obs_pair_buf = False

class Y1v0hCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1e-3
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
        run_name = 'y1v0h'
        experiment_name = 'y1v0h'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 10000
        num_steps_per_env = 24
        resume = False
        resume_path = ''
        # resume = True
        # resume_path = '/logs/y1v0h_rough/Jul09_14-01-00_y1v0h'

    