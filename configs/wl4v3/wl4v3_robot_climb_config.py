from configs.base.climb_robot_config import ClimbRobotCfg, ClimbRobotCfgPPO
class Wl4V3RobotClimbCfg( ClimbRobotCfg ):
    
    class asset( ClimbRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/wl4v3/urdf/robot.urdf'
        foot_name = "foot"
        name = "wl4v3"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( ClimbRobotCfg.rewards ):
        clearance_height_target = -0.3
        class scales( ClimbRobotCfg.rewards.scales ):

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

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.51
        max_contact_force = 250.  # forces above this value are penalized
    class terrain(ClimbRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        terrain_proportions = [0.15, 0.15, 0.0, 0.0, 0.2, 0.0, 0.0]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class Wl4V3RobotClimbCfgPPO( ClimbRobotCfgPPO ):
    class runner( ClimbRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'wl4v3_climb'
        max_iterations = 5000

 

  
