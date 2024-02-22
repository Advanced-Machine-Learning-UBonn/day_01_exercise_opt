# Understanding Autograd

This exercise studies the implementation of an Algorithmic differentiation engine via operator overloading.

Python supports overloading plus (+) and times (*) via the magic methods `__add__` and `__mul__`. Both are vital for this project.
Navigate to the `src` folder. `TODO`s mark parts of the code that require your attention.

Run `nox -s test` to check your code. If all checks pass move on to the `src/fit_neuron.py` module. Resolve all `TODO`s.


### Further reading:
- Andreas Griewank, Andrea Walther, Evaluating Derivatives
    - ( https://bonnus.ulb.uni-bonn.de/permalink/49HBZ_ULB/sol2rl/alma991037247969706467 )

- Autograd via operator overloading:
    - https://github.com/karpathy/micrograd

- Autograd via source transformation:
    - https://github.com/mattjj/autodidact/tree/master
    - https://jax.readthedocs.io/en/latest/autodidax.html
