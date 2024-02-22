"""Fit a neuron to a classification problem."""

import numpy as np
from sklearn import datasets

from autograd import ADiffFloat


def net(params, x):
    """Evaluate a small neural network."""
    hidden_weights, hidden_biases, out_weights, out_biases = params

    res = ADiffFloat(0)

    # TODO: run x through a hidden layer and an output layer.
    # Layers use the equation,
    # f(w^T x + b) .
    # Use element-wise products (*) and python's sum function
    # to compute the dot product (w^T x).
    # You will require two for loops, a first for the batch dimension and 
    # a second which loops over the individual neurons weights and biases.

    return res


if __name__ == "__main__":
    np.random.seed(42)
    size = 400
    test_size = 20
    dataset = datasets.make_classification(size, class_sep=1.5, random_state=42)

    lr = 0.1
    batch_size = 10
    epochs = 10

    batch_counter = 0

    sel = np.arange(size)
    np.random.shuffle(sel)
    train_indices = sel[: (size - test_size)]
    test_indices = sel[(size - test_size) :]

    x = dataset[0][train_indices, :]
    t = dataset[1][train_indices]
    tr_mean = np.mean(x)
    tr_std = np.std(x)
    x = (x - tr_mean) / tr_std

    in_features = 20
    hidden_neurons = 30
    out_neurons = 2

    hidden_weights = [
        [ADiffFloat(np.random.uniform(-0.5, 0.5)) for _ in range(in_features)]
        for _ in range(hidden_neurons)
    ]
    hidden_biases = [
        ADiffFloat(np.random.uniform(-0.5, 0.5)) for _ in range(hidden_neurons)
    ]

    out_weights = [
        [ADiffFloat(np.random.uniform(-0.5, 0.5)) for _ in range(hidden_neurons)]
        for _ in range(out_neurons)
    ]
    out_biases = [ADiffFloat(np.random.uniform(-0.5, 0.5)) for _ in range(out_neurons)]

    x_batched = np.split(x, len(x) // batch_size)
    t_batched = np.split(t, len(t) // batch_size)

    batch_counter = 0
    # initialize
    for e in range(epochs):
        for i in range(len(t_batched)):

            bx = x_batched[i]
            bt = t_batched[i]

            by = net((hidden_weights, hidden_biases, out_weights, out_biases), bx)

            # loop over the cost batches.
            errors = list(
                list(y - (1.0 if t == pos else 0.0) for pos, y in enumerate(y))
                for y, t in zip(by, bt)
            )
            squared_errors = (c * c for cb in errors for c in cb)
            cost = sum(squared_errors) * (1.0 / batch_size)

            cost.backward(1.0)
            print(f"e: {e:02.0f}, it: {i:02.0f}, cost value {cost.value:02.6f}")

            # update
            # TODO: Update all weights via gradient descent.
            # You need to subtract the product of learning-rate (lr) and partial
            # from every weight value.

    # testing...
    x_test = dataset[0][test_indices, :]
    t_test = dataset[1][test_indices]

    x_test = (x_test - tr_mean) / tr_std

    y_test = net((hidden_weights, hidden_biases, out_weights, out_biases), x_test)
    y_test = np.stack([[f.value for f in b] for b in y_test])
    y_test = np.argmax(y_test, -1)
    acc = sum((t_test == y_test).astype(np.float32)) / len(t_test)
    print(f"true: {t_test}")
    print(f"net:  {y_test}")
    print(f"acc: {acc}")
