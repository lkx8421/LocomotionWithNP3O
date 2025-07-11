from utils.task_registry import task_registry
from .y1v0h import *
from .y1v0h_config import *
task_registry.register("y1v0h",Y1v0h,Y1v0hCfg(),Y1v0hCfgPPO())
from .y1v0h_flat_config import *
task_registry.register("y1v0h_flat",Y1v0hFlat,Y1v0hFlatCfg(),Y1v0hFlatCfgPPO())
