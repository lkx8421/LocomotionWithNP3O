from utils.task_registry import task_registry
from .y1v0_rough_config import *
from .y1v0 import *
task_registry.register("y1v0_rough",Y1V0,Y1V0RoughCfg(),Y1V0RoughCfgPPO())  
task_registry.register("y1v0_rough_play",Y1V0,Y1V0RoughCfg_Play(),Y1V0RoughCfgPPO())  

from .y1v0_climb_config import *
from .y1v0_climb import *
task_registry.register("y1v0_climb",Y1V0Climb,Y1V0ClimbCfg(),Y1V0ClimbCfgPPO())  
task_registry.register("y1v0_climb_play",Y1V0Climb,Y1V0ClimbCfg_Play(),Y1V0ClimbCfgPPO())  

from .y1v0_flat_config import *
task_registry.register("y1v0_flat",Y1V0Flat,Y1V0FlatCfg(),Y1V0FlatCfgPPO())
task_registry.register("y1v0_flat_play",Y1V0Flat,Y1V0FlatCfg_Play(),Y1V0FlatCfgPPO())

from .y1v0_recovery_config import *
task_registry.register("y1v0_recovery",Y1V0Recovery,Y1V0RecoveryCfg(),Y1V0RecoveryCfgPPO())
task_registry.register("y1v0_recovery_play",Y1V0Recovery,Y1V0RecoveryCfg_Play(),Y1V0RecoveryCfgPPO())