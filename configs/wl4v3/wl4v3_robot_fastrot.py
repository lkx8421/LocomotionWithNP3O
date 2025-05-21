from isaacgym.torch_utils import *
import torch
# config
from configs.wl4v2.wl4v2_robot import *
class Wl4V3RobotFastrot(Wl4V2Robot):
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 1.0, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 1.0, 0., self.cfg.commands.max_curriculum)
            # self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.cfg.commands.curriculum:
            self.extras.setdefault("episode", {})
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
    # rewards
    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.all(contact, dim=1)
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05) +\
            torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05)
        return reward 
    def _reward_lin_vel_xy(self):
        return torch.sum(torch.abs(self.base_lin_vel[:, :2]), dim=1)