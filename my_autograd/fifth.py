from functools import wraps
from third import make_vjp
from third import Box
from functools import partial

def subval(values, index, replacement):
    copied = list(values)
    copied[index] = replacement
    return tuple(copied)

def grad(function, argnum=0):
    if type()
    @wraps(function)
    def gradfun(*args, **kwargs):
        def unary_function(selected):
            replaced_args = subval(args, argnum, selected)
            return function(*replaced_args, **kwargs)

        selected_value = args[argnum]
        vjp, value = make_vjp(unary_function, selected_value)
        return vjp(1.)

    return gradfun

