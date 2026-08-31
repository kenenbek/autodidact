# Autograd Learning Project: Session Handoff

## Goal and working files

The user is independently rebuilding a small reverse-mode automatic differentiation engine for learning purposes. Explain concepts and expose failure cases; do not implement the autodiff engine unless explicitly asked.

- `my_autograd/first.py`: current scalar implementation.
- `my_autograd/TODO.md`: staged learning roadmap.
- The original `autograd/` package is reference material, not the code being extended.

## Current state

Stage 2 is complete for one backward traversal. `Node.backward()` builds a topological ordering with DFS, reverses it, and correctly handles:

- A shared parent: `x * x`.
- A shared intermediate: `a + a`, where `a = x * x`.
- Branched graphs: `x*x + 3*x`.
- Deeper DAGs: `x**3 + x**2`.

The current design still combines the numerical value, graph structure, and accumulated gradient in `Node`. That is expected before stage 3.

## Known repeated-backward problem

This composite graph exposes stale intermediate gradients:

```python
x = Node(3.0)
a = x * x
y = a * x

y.backward()
assert x.grad == 27.0

y.backward()
assert x.grad == 54.0  # current code produces 72.0
```

After the first pass, `a.grad == 3`. The second call resets only `y.grad`; it does not reset `a.grad`. Backpropagating through `y = a*x` increases `a.grad` to `6`, so `a = x*x` propagates twice the required contribution. The resulting calculation is:

```text
existing x.grad                     27
direct second-pass path y -> x     +9
stale a.grad=6 through a=x*x       +18 +18
                                     -------
                                      72
```

`x ** 3` does not reveal this bug because `__pow__` creates one direct operation node and therefore has no intermediate gradient to retain.

Chosen behavior: gradients on leaf nodes accumulate across `backward()` calls, while gradients on non-leaf/intermediate nodes are temporary and must be reset before each traversal. With this behavior, two calls for the example above produce `27`, then `54`.

## Stage 3 objective

Separate values from graph metadata:

```text
Box
├── raw numerical value
└── graph Node

Node
├── parent nodes
└── stable forward recipe
```

Introduce a `primitive` decorator that:

1. Detects boxed positional arguments.
2. Extracts their raw values.
3. Calls the original numerical function.
4. Creates a graph-only `Node` containing parents and the stable forward recipe.
5. Returns the result inside a new `Box`.

Initially support only explicit `add` and `multiply` primitives. Do not add NumPy wrapping, broadcasting, higher-order differentiation, or a full VJP registry yet.

## Acceptance checks

Keep these checks focused on the current milestone:

```python
# Stage 2: shared paths and ordering.
x = Node(2.0)
a = x * x
y = a + a
y.backward()
assert x.grad == 8.0

# Repeated backward with accumulating leaf gradients.
x = Node(3.0)
a = x * x
y = a * x
y.backward()
assert x.grad == 27.0
y.backward()
assert x.grad == 54.0

# Stage 3: calls without boxes remain ordinary numerical calls.
assert add(2.0, 3.0) == 5.0

# A boxed argument triggers tracing.
x = Box(2.0, node=root)
y = add(x, 3.0)
assert isinstance(y, Box)
assert y.value == 5.0
assert y.node.parents == (root,)

# Recipes retain the values used during the forward operation.
x = Box(2.0, node=root)
y = multiply(x, x)
assert y.node.recipe.args == (2.0, 2.0)
```

When reviewing new work, distinguish intentional milestone limitations from defects. Unsupported `math.exp`, arrays, variable exponents, and constant-returning differentiation are not required for stage 3 unless the roadmap is explicitly expanded.
