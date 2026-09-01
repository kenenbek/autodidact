from first import Node
import math


def show(name, example):
    try:
        example()
    except Exception as error:
        print(f"{name}: {type(error).__name__}: {error}")


def repeated_backward():
    x = Node(3.0)
    a = x * x
    y = a * x

    y.backward()
    print("after first backward:", x.grad)

    y.backward()
    print("after second backward:", x.grad)


def mutate_after_forward():
    x = Node(2.0)
    y = x * x

    # The graph was built when x was 2, so y is still 4.
    x.value = 10.0
    y.backward()

    print("forward result:", y.value)
    print("gradient:", x.grad)


def constant_becomes_parent():
    x = Node(3.0)
    y = x + 2.0

    constant = y.parents[1]
    y.backward()

    print("number of parents:", len(y.parents))
    print("constant value:", constant.value)
    print("constant gradient:", constant.grad)


def unsupported_operation():
    x = Node(1.0)
    math.exp(x)


def constant_output():
    def f(x):
        return 7.0

    result = f(Node(3.0))
    result.backward()


# show("repeated backward", repeated_backward)
# show("mutation after forward", mutate_after_forward)
# show("constant as parent", constant_becomes_parent)
# show("math.exp", unsupported_operation)
show("constant output", constant_output)