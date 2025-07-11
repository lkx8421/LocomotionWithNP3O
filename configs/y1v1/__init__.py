from utils.task_registry import task_registry
# y1v1 task 
from .y1v1_rough_config import *
from .y1v1 import *
task_registry.register("y1v1_rough",Y1V1,Y1V1RoughCfg(),Y1V1RoughCfgPPO())