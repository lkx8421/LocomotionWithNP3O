from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *
from .base.climb_robot_config import *
from .base.climb_robot import *

from .go2.go2_robot_config import *
from .go2.go2_robot import *

from .go2w.go2w_robot_rough_config import *
from .go2w.go2w_robot import *
from .wl4v2.wl4v2_robot import *
from .wl4v2.wl4v2_robot_rough_config import *
from .tita.tita_robot import *
from .tita.tita_robot_flat_config import *

from .go2w.go2w_robot_climb_config import *
from .wl4v2.wl4v2_robot_climb_config import *
from .wl4v3.wl4v3_robot_climb_config import *

from utils.task_registry import task_registry

task_registry.register("go2",Go2Robot,Go2RobotCfg(),Go2RobotCfgPPO())
task_registry.register("tita_flat",TitaRobot,TitaRobotFlatCfg(),TitaRobotFlatCfgPPO())

task_registry.register("go2w_rough",Go2WRobot,Go2WRobotRoughCfg(),Go2WRobotRoughCfgPPO())
task_registry.register("wl4v2_rough",Wl4V2Robot,Wl4V2RobotRoughCfg(),Wl4V2RobotRoughCfgPPO())

task_registry.register("go2w_climb",ClimbRobot,Go2WRobotClimbCfg(),Go2WRobotClimbCfgPPO())
task_registry.register("wl4v2_climb",ClimbRobot,Wl4V2RobotClimbCfg(),Wl4V2RobotClimbCfgPPO())
task_registry.register("wl4v3_climb",ClimbRobot,Wl4V3RobotClimbCfg(),Wl4V3RobotClimbCfgPPO())