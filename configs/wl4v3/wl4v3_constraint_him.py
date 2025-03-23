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

from configs.wl4v2.wl4v2_constraint_him import Wl4V2ConstraintHimRoughCfg, Wl4V2ConstraintHimRoughCfgPPO

class Wl4V3ConstraintHimRoughCfg( Wl4V2ConstraintHimRoughCfg ):
    
    class asset( Wl4V2ConstraintHimRoughCfg.asset ):
        file = '{ROOT_DIR}/resources/wl4v3/urdf/robot.urdf'
        foot_name = "foot"
        name = "wl4v3"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( Wl4V2ConstraintHimRoughCfg.rewards ):
        clearance_height_target = -0.3
        class scales( Wl4V2ConstraintHimRoughCfg.rewards.scales ):

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
    class terrain(Wl4V2ConstraintHimRoughCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        terrain_proportions = [0.15, 0.15, 0.0, 0.0, 0.2, 0.0, 0.0]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class Wl4V2ConstraintHimRoughCfgPPO( Wl4V2ConstraintHimRoughCfgPPO ):
    class runner( Wl4V2ConstraintHimRoughCfgPPO.runner ):
        run_name = 'test_barlowtwins_feetcontact'
        experiment_name = 'rough_wl4v3_constraint'
        max_iterations = 5000

 

  
