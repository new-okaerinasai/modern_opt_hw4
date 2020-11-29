# from fig.hw4 import l1_ball_proj
import pytest
import numpy as np
from hw4 import run_astm, minimize, run_fw, unit_simplex_proj, bad_func, run_amd, unit_ball_proj, l1_ball_proj
import tensorflow as tf
import matplotlib.pyplot as plt


def test_dummy():
    data = tf.Variable(np.array([[10.0, 0], [0, 1.0]]))
    lm_0 = tf.Variable(np.random.randn(2))
    mean = tf.convert_to_tensor(5 * np.ones(2))
    L = np.abs(np.random.randn())
    R = 10000
    reg = lambda x: tf.reduce_sum(tf.abs(x))
    prox = lambda x: tf.reduce_sum(tf.square(x))
    func = bad_func
    R = 1000
    x = tf.convert_to_tensor(np.array([-1.0, -1.0, 4.0]))
    lm_0 = tf.Variable(np.random.randn(4))
    res = run_fw(func, lm_0, unit_simplex_proj)
    print(res)


def test_astm_lasso():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    lms = [0.01, 0.1, 1., 10., 100., 1000, 10000]
    for lm in lms:
        func = lambda x: tf.reduce_mean((tf.linalg.matvec(X_train, x) - y_train) ** 2) + lm * tf.reduce_sum(tf.abs(x))
        L = np.abs(np.random.randn())
        R = 1000
        prox = lambda x: tf.reduce_sum(tf.square(x)) / 2.
        lm_0 = tf.Variable(np.random.randn(13))
        res, loss_history = run_astm(func, lm_0, L, prox, R, max_iter=500)
        print("lambda={} RESULT = {}".format(lm, res))
        plt.figure(figsize=(12, 10))
        plt.plot(loss_history)
        plt.ylabel("loss value")
        plt.yscale("log")
        plt.xlabel("iter")
        plt.title("ASTM, $\\lambda={}$".format(lm))
        plt.show()
        plt.savefig("./astm_{}.png".format(lm))


def test_astm_bad_func():
    n = 10
    func = bad_func
    L = np.abs(np.random.randn())
    R = 1000
    prox = lambda x: tf.reduce_sum(tf.square(x)) / 2.
    lm_0 = tf.Variable(np.random.randn(n))
    x_star = tf.convert_to_tensor(np.array([1 - (i + 1) / 11 for i in range(10)]))
    optimal_value = 2 / 8 * (-1 + 1 / 11)
    gap = 3 * 2 * tf.linalg.norm(lm_0 - x_star) ** 2 / (32 * 11 ** 2)
    print(optimal_value, x_star)
    print("Gap = ", gap)
    res, loss_history = run_astm(func, lm_0, L, prox, R, max_iter=500)
    loss_history = [elem - optimal_value for elem in loss_history]
    print("RESULT = {}".format(res))
    plt.figure(figsize=(12, 10))
    plt.plot(loss_history)
    plt.ylabel("loss value")
    plt.yscale("log")
    plt.xlabel("iter")
    plt.title("ASTM, bad_func")
    plt.show()
    plt.savefig("./astm_bad_func.png")


def test_amd():
    """
        min_{x \in \Deta_n} x^TQx, s.t. min x_i \geq 0.1 / n
    """
    n = 10
    # Generate random S_n^+ matrix
    Q = tf.random.normal(shape=(n, n))
    Q = Q @ tf.transpose(Q)
    func = lambda x: tf.tensordot(x, tf.tensordot(Q, x, 1), 1)
    # min_i x_i - 0.1 / n <= 0
    g = lambda x: -tf.reduce_min(x) + 0.1 / n
    # d(x) = ||x||_2^2 / 2
    prox = lambda x: tf.reduce_sum(tf.square(x)) / 2
    # projection on \Delta_n
    proj = unit_simplex_proj
    # x_0 = argmin_{x \in \Delta_n} d(x)
    x0 = tf.Variable(proj(tf.zeros(n)))
    result, loss_history = run_amd(func, g, x0, prox, proj, eps=1e-6, max_iter=100)
    loss_history = [elem - optimal_value for elem in loss_history]
    plt.plot(loss_history)
    plt.ylabel("loss value")
    plt.yscale("log")
    plt.xlabel("iter")
    plt.title("Adaptive Mirror Descent")
    plt.savefig("./amd.png")
    print(result)

def test_fw_bad_func():
    n = 10
    func = bad_func
    L = np.abs(np.random.randn())
    R = 1000
    prox = lambda x: tf.reduce_sum(tf.square(x)) / 2.
    lm_0 = tf.Variable(np.random.randn(n))
    x_star = tf.convert_to_tensor(np.array([1 - (i + 1) / 11 for i in range(10)]))
    optimal_value = 2 / 8 * (-1 + 1 / 11)
    gap = 3 * 2 * tf.linalg.norm(lm_0 - x_star) ** 2 / (32 * 11 ** 2)
    print(optimal_value, x_star)
    print("Gap = ", gap)
    proj = unit_simplex_proj
    x0 = tf.Variable(proj(tf.zeros(n)))
    result, loss_history = run_fw(func, x0, proj)
    loss_history = [elem - optimal_value for elem in loss_history]
    plt.plot(loss_history)
    plt.ylabel("loss value")
    plt.yscale("log")
    plt.xlabel("iter")
    plt.title("FW")
    plt.savefig("./fw_bad.png")
    print(result)    

def test_fw_lasso():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    lms = [1., 10., 100., 1000, 10000]
    for lm in lms:
        func = lambda x: tf.reduce_mean((tf.linalg.matvec(X_train, x) - y_train) ** 2)
        proj = lambda x: l1_ball_proj(x, R=lm)
        lm_0 = tf.Variable(np.random.randn(13))
        res, loss_history = run_fw(func, lm_0, proj, max_iter=500)
        print("lambda={} RESULT = {}".format(lm, res))
        plt.figure(figsize=(12, 10))
        plt.plot(loss_history)
        plt.ylabel("loss value")
        plt.yscale("log")
        plt.xlabel("iter")
        plt.title("Frank-Wolfe, $\\R={}$".format(lm))
        plt.show()
        plt.savefig("./fw_lasso_{}.png".format(lm))  
        print("RESULT=", res)


if __name__ == "__main__":
    test_fw_lasso(