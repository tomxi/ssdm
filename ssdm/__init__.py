import numpy as np
np.int = int
np.float = float

from .core import *
from .formatting import *
from .utils import *

# from . import feature
# from . import scluster

from . import viz
viz.hv.extension('bokeh')

from . import harmonix as hmx
from . import salami as slm
from . import jsd
from . import rwcpop
from . import spam

from . import scanner as scn

from .expand_hier import expand_hierarchy
