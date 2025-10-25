# omode

omode is a generic modeler for optimization problems in Python. It consists of a thin wrapper around existing modeling frameworks such as CasADi and cvxpy, providing a unified syntax for building and solving optimization problems.

---

## Installation

```bash
pip install omode
```
*(Install solver backends like `casadi`, `cvxpy` separately as needed.)*

## Quick Example

```python
import omode

model = omode.Model()
x = model.add_variable('x')
y = model.add_variable('y')

model.set_objective(x**2 + y**2)
model.add_constraint(x + y == 1)
model.add_constraint(x >= 0)
model.add_constraint(y >= 0)

result = model.solve(backend='casadi')
print(result)
```

## Currently supported backends

- CasADi
- cvxpy


## License

MIT © [Mathijssch](https://github.com/Mathijssch)
