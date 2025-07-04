import warnings
import torch # required to force load shared libraries
if torch.cuda.is_available():
    try:
        from .impl.cuda.ops import linrec_pipe as linrec
    except ImportError:
        warnings.warn("linrec.impl.cuda not found, resorting to linrec.impl.triton.")
        from .impl.triton.ops import linrec_pipe as linrec
else:
    warnings.warn("cuda is not available, resorting to linrec.impl.python.")
    from .impl.python.ops import linrec_ref as linrec   
