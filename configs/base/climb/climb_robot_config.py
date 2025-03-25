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

class ClimbRobotCfg( LeggedRobotCfg ):
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
        """
          unitree go2 sdk order:
               -0.1 <-3 FR_hip_joint 0 -> 0.0
               0.8 <- 4 FR_thigh_joint 1 -> 0.9
               -1.5 <- 5 FR_calf_joint 2 -> -1.8
               0.1 <- 0 FL_hip_joint 3 -> 0.0
               0.8 <- 1 FL_thigh_joint 4 -> 0.9
               -1.5 <- 2 FL_calf_joint 5 -> -1.8
               -0.1 <- 9 RR_hip_joint 6 -> 0.0
               1 <- 10 RR_thigh_joint 7 -> 0.9
               -1.5 <- 11 RR_calf_joint 8 -> -1.8
               0.1 <- 6 RL_hip_joint 9 -> 0.0
               1 <- 7 RL_thigh_joint 10 -> 0.9
               -1.5 <- 8 RL_calf_joint 11 -> -1.8
        """
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.0,     # [rad]
            'FR_thigh_joint': 0.0,     # [rad]
            'RL_thigh_joint': 0.6,   # [rad]
            'RR_thigh_joint': 0.6,   # [rad]

            'FL_calf_joint': -0.9,   # [rad]
            'FR_calf_joint': -0.9,  # [rad]
            'RL_calf_joint': -1.2,    # [rad]
            'RR_calf_joint': -1.2,    # [rad]

            'FL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

        start_joint_angles = { # = target angles [rad] when stand still
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.6,     # [rad]
            'RL_thigh_joint': 0.6,   # [rad]
            'FR_thigh_joint': 0.6,     # [rad]
            'RR_thigh_joint': 0.6,   # [rad]

            'FL_calf_joint': -1.2,   # [rad]
            'RL_calf_joint': -1.2,    # [rad]
            'FR_calf_joint': -1.2,  # [rad]
            'RR_calf_joint': -1.2,    # [rad]

            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 30.,
                     'thigh': 30.,
                     'calf': 30.,
                     'foot': 2.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5

        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/wl4ver2/urdf/robot.urdf'
        foot_name = "foot"
        name = "wl4v2"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        clearance_height_target = -0.3
        class scales( LeggedRobotCfg.rewards.scales ):

            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2 # -0
            torques = -1e-5 # -0.00001
            dof_pos_limits = -50.0
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -1.0 # 0
            feet_air_time = 0.0 # 1
            collision = -1.0
            # base_collision = -50.0
            feet_stumble = 0.0
            action_rate = -0.002
            stand_still = 0.10
            feet_all_contact = 0.1
            powers = -2e-5
            action_smoothness= -0.001

            foot_mirror = 0.15
            hip_pos = 0.25
            foot_swing_clearance = -0.0
            climb_pitch = 1.0
            climb_feet_air = 0.5
            # climb_feet_lift= 0.5

            heading = -0.05
            # com_feet_contact = 0.4
            # foot_clearance = -0.01
            # feet_relative_x = 0.1
            # contact_body_pitch = 1.2
            # front_feet_air = 1.0
            # feet_upper_height = 2.0
            # stand_joint_pos = 0.8
            # stand_height = 0.5
            
            # feet_lin_pos_z = 1.2
            # position_tracking = 1.5
            # feet_lin_vel_z = 10.0
            # feet_height = 2.0 # 有点上坡
            # climbing_50cm = 1.5
            # front_feet_air = -1.0
            # foot_clearance= -0.0
            # front_rear_feet_air = 0.1
            # # two wheel stand
            # front_feet_air = 1
            # head_pitch = 2

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.51
        max_contact_force = 250.  # forces above this value are penalized

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 2.75]
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

        randomize_kpkd = False
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        # randomize_initial_joint_pos = True
        # initial_joint_pos_range = [0.5, 1.5]
    
    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class costs:
        num_costs = 3
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1
            #foot_slide = 1
            #foot_nocontact_regular = 1
            # feet_air_time = 1
            # foot_mirror = -0.1
            # trot_contact=0.1
            # stand_still=0.1
            # #idol_contact = 0.1
            # #idol_contact = 0.1
            #base_height = 0.1
            # foot_swing_clearance = 1
            #acc_smoothness = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            #foot_slide = 0.0
            #foot_nocontact_regular = 0.0
            #feet_air_time = 0.0
            #foot_mirror = 2.0
            # trot_contact= 5.0
            # stand_still = 0.0
            # #idol_contact = 0.0
            # #idol_contact = 0.0
            #base_height = 0.0
            # foot_swing_clearance = 0.0
            #acc_smoothness = 2.0
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        terrain_proportions = [0.15, 0.15, 0.0, 0.0, 0.2, 0.0, 0.0]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class ClimbRobotCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'climb_robot'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 5000
        num_steps_per_env = 24
        resume = False
        resume_path = ''
 

  
