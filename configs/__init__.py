from .base.base_config import *
from .base.base_task import *
from .base.legged_robot_config import *
from .base.legged_robot import *

from .go2.go2_constraint_him import *

from .wl4v2.wl4v2_constraint_him import *
from .wl4v2.wl4v2_legged_robot import *

from .wl4v3.wl4v3_constraint_him import *
from .wl4v3.wl4v3_legged_robot import *

from .go2w.go2w_constraint_him import *
from .go2w.go2w_legged_robot import *

from utils.task_registry import task_registry

task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
task_registry.register("wl4v2",Wl4V2LeggedRobot,Wl4V2ConstraintHimRoughCfg(),Wl4V2ConstraintHimRoughCfgPPO())
task_registry.register("wl4v3",Wl4V3LeggedRobot,Wl4V3ConstraintHimRoughCfg(),Wl4V3ConstraintHimRoughCfgPPO())
task_registry.register("go2w",Go2WLeggedRobot,Go2WConstraintHimRoughCfg(),Go2WConstraintHimRoughCfgPPO())