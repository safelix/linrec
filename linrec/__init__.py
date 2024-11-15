import torch # required to force load shared libraries
from . import _C
from .impl.cuda.ops import linrec_pipe as linrec