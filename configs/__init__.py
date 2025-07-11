from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *
# from .go2.go2_robot_config import *
# from .go2.go2_robot import *
from utils.task_registry import task_registry
# task_registry.register("go2",Go2Robot,Go2RobotCfg(),Go2RobotCfgPPO())

# from .go2w.go2w_robot_rough_config import *
# from .go2w.go2w_robot import *
# task_registry.register("go2w_rough",Go2WRobot,Go2WRobotRoughCfg(),Go2WRobotRoughCfgPPO())
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

# y1v0h task 
from .y1v0h.y1v0h import *
from .y1v0h.y1v0h_config import *
task_registry.register("y1v0h",Y1v0h,Y1v0hCfg(),Y1v0hCfgPPO())
from .y1v0h.y1v0h_flat_config import *
task_registry.register("y1v0h_flat",Y1v0hFlat,Y1v0hFlatCfg(),Y1v0hFlatCfgPPO())


