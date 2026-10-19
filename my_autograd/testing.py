from third import Box, Node
from third import multiply, add
from third import trace
from third import get_vjp_rule, primitive_vjps, defvjp
from third import make_vjp


def test_get_vjp_rule_for_multiply():
    left_rule = get_vjp_rule(multiply, 0)
    right_rule = get_vjp_rule(multiply, 1)

    assert left_rule is primitive_vjps[multiply][0]
    assert right_rule is primitive_vjps[multiply][1]

    assert left_rule(1.0, 6.0, 2.0, 3.0) == 3.0
    assert right_rule(1.0, 6.0, 2.0, 3.0) == 2.0


def test_trace_connected_output():
    value, end_node, start_node = trace(lambda x: x*x, 3.0)

    assert value == 9
    assert end_node is not None
    assert start_node.recipe is None
    assert end_node.recipe.function is multiply


def test_trace_constant_output():
    value, end_node, start_node = trace(lambda x: 7.0, 3.0)

    assert value == 7.0
    assert end_node is None
    assert start_node.recipe is None


def test_make_vjp_runs_forward_once():
    calls = 0

    def function(x):
        nonlocal calls
        calls += 1
        return x*x

    vjp, result = make_vjp(function, 3.0)

    assert calls == 1
    assert result == 9.0

    vjp(1.0)
    vjp(2.0)

    assert calls == 1


def test_seed_scales_vjp():
    vjp, result = make_vjp(lambda x: x*x, 3.0)

    assert result == 9.0
    assert vjp(1.0) == 6.0
    assert vjp(2.0) == 12.0
    assert vjp(-1.0) == -6.0


def test_argument_positions_are_recorded():
    left_vjp, _ = make_vjp(lambda x: x*3.0, 2.0)
    right_vjp, _ = make_vjp(lambda x: x*3.0, 2.0)

    assert left_vjp(1.0) == 3.0
    assert right_vjp(1.0) == 3.0

    root = Node.new_root()
    x = Box(2.0, root)

    left = multiply(x, 3.0)
    right = multiply(3.0, x)

    assert left.node.recipe.argnums == (0, )
    assert right.node.recipe.argnums == (1, )
    assert left.node.parents == (root, )
    assert right.node.parents == (root, )


def test_shared_parent():
    vjp, _ = make_vjp(lambda x: x*x, 3.0)
    assert vjp(1.0) == 6.0

def test_branch_merge():
    vjp, _ = make_vjp(lambda x: x*x+x, 3.0)
    assert vjp(1.0) == 7.0

def test_deep_graph():
    vjp, _ = make_vjp(lambda x: x*x*x, 3.0)
    assert vjp(1.0) == 27.0


def test_vjp_has_no_stale_intermediate_state():
    vjp, _ = make_vjp(lambda x:x*x*x, 3.0)

    first = vjp(1.0)
    second = vjp(1.0)
    scaled = vjp(2.0)

    assert first == 27.0
    assert second == 27.0
    assert scaled == 54.0


def test_constant_function_has_zero_vjp():
    vjp, result = make_vjp(lambda x: 7.0, 3.0)

    assert result == 7.0
    assert vjp(1.0) == 0.0
    assert vjp(100.0) == 0.0

def test_missing_primitive_rule_is_clear():
    @primitive
    def subtract(x, y):
        return x - y
    vjp, _ = make_vjp(lambda x: subtract(x, 1.0), 3.0)
    try:
        vjp(1.0)
    except NotImplementedError as error:
        assert "subtract" in str(error)
        assert "argument 0" in str(error)
    else:
        raise AssertionError("Expected a missing-VJP error")


if __name__ == '__main__':
    test_shared_parent()
    test_branch_merge()
    test_deep_graph()

