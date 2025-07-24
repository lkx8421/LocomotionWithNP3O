from isaacgym.torch_utils import *
import torch
# env related
from .y1v0 import *
import numpy as np
from isaacgym import gymtorch

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

class Y1V0Climb( Y1V0 ):
    def _init_buffers(self):
        super()._init_buffers()
        self.front_climb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.rear_climb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.random_angle = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.in_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def _post_physics_step_callback(self):
        # self.dof_pos[:, self.foot_joint_indices] = 0
        self._update_climb_condition()
        super()._post_physics_step_callback()
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        rot_mat = quat_to_rot_matrix(self.base_quat)
        self.reset_buf |= rot_mat[:, 2, 2] < -0.5

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
        self.rear_climb = torch.logical_and(bool_rear, torch.any(feet_pos_z[:, 2:4] < -0.01, dim=1)) * ~self.front_climb
        # update pit
        # if self.cfg.terrain.curriculum:
        self.in_pit = self.terrain_types/self.cfg.terrain.num_cols >= sum(self.cfg.terrain.terrain_proportions)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # is_pit = self.terrain_types[env_ids]/self.cfg.terrain.num_cols >= sum(self.cfg.terrain.terrain_proportions)

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0] * ~is_pit, self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0] * ~is_pit, self.command_ranges["lin_vel_y"][1] * ~is_pit, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] = self.commands[env_ids, 0] * ~(self.in_pit[env_ids] & (self.commands[env_ids, 0] < 0))
        self.commands[env_ids, 1] = self.commands[env_ids, 1] * ~(self.in_pit[env_ids])
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            ## 
            reference_angles = torch.tensor([0, np.pi/2, -np.pi/2, np.pi], device=self.device)
            diffs = torch.abs(self.random_angle[env_ids].unsqueeze(1) - reference_angles)
            closest_indices = torch.argmin(diffs, dim=1)
            random_indices = torch.randint(0, len(reference_angles), (len(env_ids),), device=self.device)
            self.commands[env_ids, 3] = torch.where(self.in_pit[env_ids], reference_angles[closest_indices], self.commands[env_ids, 3])# reference_angles[closest_indices] * ~(is_pit)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation
        self.random_angle[env_ids] = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        # random_axis = torch_rand_float(-1., 1., (len(env_ids), 3), device=self.device)
        # random_axis = normalize(random_axis)  # 归一化为单位向量
        random_axis = to_torch([0., 0., 1.], device=self.device).repeat((len(env_ids), 1))
        random_rotation = quat_from_angle_axis(self.random_angle[env_ids], random_axis)
        # print("a", self.random_angle)
        self.root_states[env_ids, 3:7] = random_rotation
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) * ~self.in_pit
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * ~self.in_pit
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        base_x_axis = torch.stack([
            1 - 2*self.base_quat[:, 1]**2 - 2*self.base_quat[:, 2]**2, 
            2*self.base_quat[:, 0]*self.base_quat[:, 1] + 2*self.base_quat[:, 3]*self.base_quat[:, 2], 
            2*self.base_quat[:, 0]*self.base_quat[:, 2] - 2*self.base_quat[:, 3]*self.base_quat[:, 1]
        ], dim=1).to(self.device)
        dot_product = torch.clip(torch.sum(base_x_axis * torch.tensor([0, 0, 1], device=self.device), dim=-1), -1, 1)
        angle_error = torch.acos(dot_product)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)# * (angle_error > 0.4*torch.pi) #* ~self.front_climb

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
        reward_front_pitch = -torch.square(base_x_world_z_angle - np.pi/8) # 0.25 * torch.exp(-torch.square(base_x_world_z_angle)/0.5) 
        reward_rear_pitch = -torch.square(base_z_world_z_angle) # 0.25 * torch.exp(-torch.square(base_z_world_z_angle)/0.5) 
        reward_front = torch.where(self.front_climb, reward_front_pitch, torch.zeros_like(reward_front_pitch))
        reward_rear = torch.where(self.rear_climb, reward_rear_pitch, torch.zeros_like(reward_rear_pitch))
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
