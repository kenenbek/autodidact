from functools import wraps
from third import make_vjp
from third import Box
from functools import partial

def subval(values, index, replacement):
    copied = list(values)
    copied[index] = replacement
    return tuple(copied)

def grad(function, argnum=0):
    if type(argnum) is not int:
        raise TypeError(f"argnum must be an int, got {type(argnum).__name__}")
    if argnum < 0:
        raise ValueError(f"argnum must be nonnegative, got {argnum}")
    @wraps(function)
    def gradfun(*args, **kwargs):
        if argnum >= len(args):
            raise ValueError(
                f"argnum={argnum} must be passed positionally, but only "
                f"{len(args)} positional argument(s) were given"
            )
        def unary_function(selected):
            replaced_args = subval(args, argnum, selected)
            return function(*replaced_args, **kwargs)

        selected_value = args[argnum]
        vjp, value = make_vjp(unary_function, selected_value)
        return vjp(1.)

    return gradfun

