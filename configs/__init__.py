from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *
from .base.climb.climb_robot_config import *
from .base.climb.climb_robot import *

from .go2.go2_robot_config import *
from .go2.go2_robot import *

from .go2w.go2w_constraint_him import *
from .go2w.go2w_legged_robot import *
from .wl4v2.wl4v2_robot import *
from .wl4v2.wl4v2_robot_config import *
from .tita.tita_robot import *
from .tita.tita_robot_config import *

from .go2w.climb.go2w_climb_robot_config import *
from .wl4v2.climb.wl4v2_climb_robot_config import *
from .wl4v3.climb.wl4v3_climb_robot_config import *

from utils.task_registry import task_registry

task_registry.register("go2",Go2Robot,Go2RobotCfg(),Go2RobotCfgPPO())
task_registry.register("go2w",Go2WLeggedRobot,Go2WConstraintHimRoughCfg(),Go2WConstraintHimRoughCfgPPO())
task_registry.register("wl4v2",Wl4V2Robot,Wl4V2RobotCfg(),Wl4V2RobotCfgPPO())
task_registry.register("tita",TitaRobot,TitaRobotCfg(),TitaRobotCfgPPO())

task_registry.register("go2w_climb",ClimbRobot,Go2WClimbRobotCfg(),Go2WClimbRobotCfgPPO())
task_registry.register("wl4v2_climb",ClimbRobot,Wl4V2ClimbRobotCfg(),Wl4V2ClimbRobotCfgPPO())
task_registry.register("wl4v3_climb",ClimbRobot,Wl4V3ClimbRobotCfg(),Wl4V3ClimbRobotCfgPPO())