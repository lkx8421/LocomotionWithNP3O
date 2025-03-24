from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *

from .go2.go2_constraint_him import *
from .go2.go2_legged_robot import *

from .climb_base.climb_robot_config import *
from .climb_base.climb_robot import *
from .wl4v2_climb.wl4v2_climb_robot_config import *
from .wl4v3_climb.wl4v3_climb_robot_config import *

from .go2w.go2w_constraint_him import *
from .go2w.go2w_legged_robot import *

from utils.task_registry import task_registry

task_registry.register("go2",Go2LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())

task_registry.register("go2w",Go2WLeggedRobot,Go2WConstraintHimRoughCfg(),Go2WConstraintHimRoughCfgPPO())
task_registry.register("wl4v2_climb",ClimbRobot,Wl4V2ClimbRobotCfg(),Wl4V2ClimbRobotCfgPPO())
task_registry.register("wl4v3_climb",ClimbRobot,Wl4V3ClimbRobotCfg(),Wl4V3ClimbRobotCfgPPO())