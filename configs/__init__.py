from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *
from .go2.go2_robot_config import *
from .go2.go2_robot import *
from utils.task_registry import task_registry
task_registry.register("go2",Go2Robot,Go2RobotCfg(),Go2RobotCfgPPO())

from .go2w.go2w_robot_rough_config import *
from .go2w.go2w_robot import *
task_registry.register("go2w_rough",Go2WRobot,Go2WRobotRoughCfg(),Go2WRobotRoughCfgPPO())
# tita task 
from .tita.tita import *
from .tita.tita_flat_config import *
task_registry.register("tita_flat",Tita,TitaFlatCfg(),TitaFlatCfgPPO())

# y1v0 task 
from .y1v0.y1v0_rough_config import *
from .y1v0.y1v0 import *
task_registry.register("y1v0_rough",Y1V0,Y1V0RoughCfg(),Y1V0RoughCfgPPO())  

# y1v1 task 
from .y1v1.y1v1_rough_config import *
from .y1v1.y1v1 import *
task_registry.register("y1v1_rough",Y1V1,Y1V1RoughCfg(),Y1V1RoughCfgPPO())



from .wl4v2.wl4v2_robot_climb import *
from .wl4v2.wl4v2_robot_climb_config import *
from .wl4v3.wl4v3_robot_climb_config import *

from .wl4v2.wl4v2_robot import *
from .wl4v2.wl4v2_robot_rough_config import *
from .wl4v2.wl4v2_robot_crazy_config import *

task_registry.register("wl4v2_rough",Wl4V2Robot,Wl4V2RobotRoughCfg(),Wl4V2RobotRoughCfgPPO())
task_registry.register("wl4v2_crazy",Wl4V2Robot,Wl4V2RobotCrazyCfg(),Wl4V2RobotCrazyCfgPPO())

# task_registry.register("go2w_climb",ClimbRobot,Go2WRobotClimbCfg(),Go2WRobotClimbCfgPPO())
task_registry.register("wl4v2_climb",Wl4V2ClimbRobot,Wl4V2RobotClimbCfg(),Wl4V2RobotClimbCfgPPO())
task_registry.register("wl4v3_climb",Wl4V2ClimbRobot,Wl4V3RobotClimbCfg(),Wl4V3RobotClimbCfgPPO())

from .wl4v3.wl4v3_robot_scarp_config import *
task_registry.register("wl4v3_scarp",Wl4V3RobotScarp,Wl4V3RobotScarpCfg(),Wl4V3RobotScarpCfgPPO())

from .wl4v3.wl4v3_robot_fastrot_config import *
task_registry.register("wl4v3_fastrot",Wl4V3RobotFastrot,Wl4V3RobotFastrotCfg(),Wl4V3RobotFastrotCfgPPO())

from .wl4v3.wl4v3_robot_hopturn_config import *
from .wl4v3.wl4v3_robot_hopturn import *
task_registry.register("wl4v3_hopturn",Wl4V3RobotHopturn,Wl4V3RobotHopturnCfg(),Wl4V3RobotHopturnCfgPPO())

