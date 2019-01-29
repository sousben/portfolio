import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


n_feature = 2
n_class = 2


def make_network(n_hidden=100):
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )

    return model


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'])

    return h, prob


def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs

    dh = errs @ model['W2'].T
    dh[hs < 0] = 0
    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)


def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        err = y_true - y_pred

        xs.append(x)
        hs.append(h)
        errs.append(err)

    return backward(model, np.array(xs), np.array(hs), np.array(errs))


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


def sgd(model, X_train, y_train, minibatch_size):
    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        # Provided
        for layer in grad:
            model[layer] += learning_rate * grad[layer]

    return model


def momentum(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(val) for k, val in model.items()}
    mu = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = mu * velocity[layer] + learning_rate * grad[layer]
            model[layer] += velocity[layer]

    return model


def nesterov(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        model_ahead = {k: mod + gamma * velocity[k] for k, mod in model.items()}
        grad = get_minibatch_grad(model_ahead, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + learning_rate * grad[layer]
            model[layer] += velocity[layer]

    return model


def adagrad(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(val) for k, val in model.items()}

    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    grad_squared = cache

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)
        grad_squared = {k: grad_squared[k] + gr**2 for k, gr in grad.items()}

        for k in grad:
            model[k] += learning_rate * grad[k] / (np.sqrt(grad_squared[k]) + 1e-7)

    return model


def rmsprop(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(val) for k, val in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)
    grad_squared = cache

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)
        grad_squared = {k: gamma * grad_squared[k] + (1 - gamma) * gr ** 2 for k, gr in grad.items()}

        for k in grad:
            model[k] += learning_rate * grad[k] / (np.sqrt(grad_squared[k]) + 1e-7)

    return model


def adam(model, X_train, y_train, minibatch_size):
    M = {k: np.zeros_like(v) for k, v in model.items()}
    R = {k: np.zeros_like(v) for k, v in model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            M[k] = beta1 * M[k] - (1. - beta1) * grad[k]
            R[k] = beta2 * R[k] + (1. - beta2) * grad[k]**2

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            model[k] -= learning_rate * m_k_hat / (np.sqrt(r_k_hat) + 1e-7)

    return model


def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]


if __name__ == '__main__':
    X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    n_iter = 100
    eps = 1e-8  # Smoothing to avoid division by zero
    learning_rate = 1e-2
    minibatch_size = 100
    n_experiment = 3

    algos = dict(
        sgd=sgd,
        momentum=momentum,
        nesterov=nesterov,
        adagrad=adagrad,
        rmsprop=rmsprop,
        adam=adam
    )

    algo_accs = {k: np.zeros(n_experiment) for k in algos}

    for algo_name, algo in algos.items():
        print('Experimenting on {}'.format(algo_name))

        for k in range(n_experiment):
            # print('Experiment-{}'.format(k))

            # Reset model
            model = make_network()
            model = algo(model, X_train, y_train, minibatch_size)

            y_pred = np.zeros_like(y_test)

            for i, x in enumerate(X_test):
                _, prob = forward(x, model)
                y = np.argmax(prob)
                y_pred[i] = y

            algo_accs[algo_name][k] = np.mean(y_pred == y_test)

    print()

    for k, v in algo_accs.items():
        print('{} => mean accuracy: {}, std: {}'.format(k, v.mean(), v.std()))