import numpy as np

np.int = int
np.float = float

from . import base
from .configs import *
from .formatting import *
from .utils import *

# from . import feature
# from . import scluster

from . import viz

viz.hv.extension("bokeh")

from .corpus import hmx, rwcpop, slm, jsd

from . import scanner as scn

from .expand_hier import expand_hierarchy
