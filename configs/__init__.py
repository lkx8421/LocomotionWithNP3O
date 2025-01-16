from .base_config import *
from .legged_robot_config import *
from envs import LeggedRobot
from utils.task_registry import task_registry

from .go2.go2_constraint_him import *
from .wl4.wl4_constraint_him import *
from .wl4.wl4_legged_robot import *

task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
task_registry.register("wl4",Wl4LeggedRobot,Wl4ConstraintHimRoughCfg(),Wl4ConstraintHimRoughCfgPPO())
