from utils.task_registry import task_registry
from .y1v0_rough_config import *
from .y1v0 import *
task_registry.register("y1v0_rough",Y1V0,Y1V0RoughCfg(),Y1V0RoughCfgPPO())  
task_registry.register("y1v0_rough_play",Y1V0,Y1V0RoughCfg_Play(),Y1V0RoughCfgPPO())  

from .y1v0_climb_config import *
from .y1v0_climb import *
task_registry.register("y1v0_climb",Y1V0Climb,Y1V0ClimbCfg(),Y1V0ClimbCfgPPO())  
task_registry.register("y1v0_climb_play",Y1V0Climb,Y1V0ClimbCfg_Play(),Y1V0ClimbCfgPPO())  
