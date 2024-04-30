import numpy as np
np.int = int
np.float = float

from .core import *
from .formatting import *
from .utils import *
from .formatting import *

# from . import feature
# from . import scluster
# from . import scanner
from . import viz
viz.hv.extension('bokeh')

from . import harmonix as hmx
from . import salami as slm
from . import jsd

from .expand_hier import expand_hierarchy
