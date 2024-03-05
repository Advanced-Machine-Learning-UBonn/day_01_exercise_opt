# Understanding Autograd

Thank you for taking our course. Completing the following tasks will prepare you for the exercise sessions in the coming weeks.
Consequently, we will use workstations running Ubuntu Linux. We highly recommend to use Linux systems instead of Windows.

If you are unfamiliar with GitHub or our exercise setup you can follow Task 1 - 5 [here](https://github.com/Deep-Learning-with-Jax/day_01_exercise_intro).

## Task 1: Operator overloading
This exercise studies the implementation of an Algorithmic differentiation engine via operator overloading.

Python supports overloading plus (+) and times (*) via the magic methods `__add__` and `__mul__`. Both are vital for this project.
Navigate to the `src` folder. `TODO`s mark parts of the code that require your attention.

Run `nox -s test` to check your code. If all checks pass move on to the `src/fit_neuron.py` module. Resolve all `TODO`s.
To do so recall that the multivariate chain rule requires as to sum up contributions from each path. More formally for an input $x_j$ we compute,

$$
    \delta x_j = \sum_i \frac{\partial y_i}{\partial x_j} \delta y_i.
$$

$\frac{\partial y_i}{\partial x_j}$ can be a chain of multiple elemental operations.

When overloading `__add__` please consider,

$$
    \begin{align}
        y = x_1 + x_2 & \\ 
        & \rightarrow \delta x_1 = \frac{\partial (x_1 + x_2)}{\partial x_1} \cdot \delta y = 1 \cdot \delta y \\
        & \rightarrow \delta x_2 = \frac{\partial (x_1 + x_2)}{\partial x_2} \cdot \delta y = 1 \cdot \delta y ,
    \end{align}
$$

with $\delta y$ as the inner derivative or seed value.

When overloading `__mul__` please consider,

$$
    \begin{align}
        y = x_1 \cdot x_2 & \\ 
        & \rightarrow \delta x_1 = \frac{\partial (x_1 \cdot  x_2)}{\partial x_1} \cdot \delta y = x_2 \cdot \delta y \\
        & \rightarrow \delta x_2 = \frac{\partial (x_1 \cdot  x_2)}{\partial x_2} \cdot \delta y = x_1 \cdot \delta y.
    \end{align}
$$

Finally for element-wise functions

$$
   \begin{align}
        y = f(x) & \\
                 & \rightarrow \delta x = f'(x)\delta y .
   \end{align}
$$

### Optional further reading:
- Andreas Griewank, Andrea Walther, Evaluating Derivatives
    - ( https://bonnus.ulb.uni-bonn.de/permalink/49HBZ_ULB/sol2rl/alma991037247969706467 )

- Autograd via operator overloading:
    - https://github.com/karpathy/micrograd

- Autograd via source transformation:
    - https://github.com/mattjj/autodidact/tree/master
    - https://jax.readthedocs.io/en/latest/autodidax.html

- Wikipedia's article on Automatic Differentiation
    - https://en.wikipedia.org/wiki/Automatic_differentiation
