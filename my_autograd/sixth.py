import numpy as onp
from my_autograd.third import make_vjp
from functools import wraps
from fifth import subval

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
        vjp, result = make_vjp(unary_function, selected_value)
        if onp.shape(result) != ():
            raise ValueError(
                        f"grad only supports scalar-output functions, got output with shape {result.shape}"
            )
        return vjp(onp.ones_like(result))

    return gradfun



if __name__ == '__main__':
    x = np.array([1., 2.])
    vjp, result = make_vjp(lambda x: x*x, x)
    np.testing.assert_allclose(result, [1., 4.])
    np.testing.assert_allclose(vjp(np.array([3., 4.])), [6., 16.])
    print(vjp(np.array([3., 4.])))