# Understanding Autograd

This exercise studies the implementation of an Algorithmic differentiation engine via operator overloading.

Python supports overloading plus (+) and times (*) via the magic methods `__add__` and `__mul__`. Both are vital for this project.
Navigate to the `src` folder. `TODO`s mark parts of the code that require your attention.

Run `nox -s test` to check your code. If all checks pass move on to the `src/fit_neuron.py` module. Resolve all `TODO`s.

``` math
    \delta x_j = \sum_i \frac{\partial y_i}{\partial x_j} \delta y_i.
```

``` math
    \begin{align}
        y = x_1 + x_2 & \\ 
        & \rightarrow \delta x_1 = \frac{\partial (x_1 + x_2)}{\partial x_1} \cdot \delta y = 1 \cdot \delta y \\
        & \rightarrow \delta x_2 = \frac{\partial (x_1 + x_2)}{\partial x_2} \cdot \delta y = 1 \cdot \delta y
    \end{align}
```


```math
    \begin{align}
        y = x_1 \cdot x_2 & \\ 
        & \rightarrow \delta x_1 = \frac{\partial (x_1 \cdot  x_2)}{\partial x_1} \cdot \delta y = x_2 \cdot \delta y \\
        & \rightarrow \delta x_2 = \frac{\partial (x_1 \cdot  x_2)}{\partial x_2} \cdot \delta y = x_1 \cdot \delta y,
    \end{align}
```
with $\delta y$ as the inner derivative or seed value.


```math
   \begin{align}
        y = f(x) & \\
                 & \rightarrow \delta x = f'(x)\delta y .
   \end{align}
```

### Further reading:
- Andreas Griewank, Andrea Walther, Evaluating Derivatives
    - ( https://bonnus.ulb.uni-bonn.de/permalink/49HBZ_ULB/sol2rl/alma991037247969706467 )

- Autograd via operator overloading:
    - https://github.com/karpathy/micrograd

- Autograd via source transformation:
    - https://github.com/mattjj/autodidact/tree/master
    - https://jax.readthedocs.io/en/latest/autodidax.html
