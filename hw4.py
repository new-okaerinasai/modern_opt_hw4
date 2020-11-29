import numpy as np
# from scipy.optimize import minimize
import logging
import tensorflow as tf
import tqdm

def solve_quadratic(a, b, c):
    """
    Solve arbitrary quadratic equation ax^2 + bx + c = 0
    """
    if a == 0:
        if b == 0:
            if c != 0:
                raise RuntimeError("no roots in the equation")
            else:
                return 0, 0
        else:
            return -c / b, -c / b
    else:
        d = b ** 2 - 4 * a * c
        if d >= 0:
            return (-b - np.sqrt(d)) / (2 * a), (-b + np.sqrt(d)) / (2 * a)
        else:
            raise RuntimeError("no roots in the equation")


def minimize(func, x_0, proj=None):
    """
        Run Adam optimizer to find minimum of a function `func`
         with the initial point `x_0`.
        proj: callable. Function of projection on some set.
    """
    optimizer = tf.optimizers.Adam(1e-1)
    loss_prev = 1e+12
    for i in range(100):
        with tf.GradientTape() as tape:
            tape.watch(x_0)
            loss = func(x_0)
        gradients = tape.gradient(loss, x_0)
        optimizer.apply_gradients(zip([gradients], [x_0]))
        if proj is not None:
            x_0 = tf.Variable(proj(x_0))
        if tf.abs(loss - loss_prev) < 1e-5 * loss:
            break
        loss_prev = loss
    return x_0


def unit_simplex_proj(x):
    """
    Function of projection on the unit simplex in R^n
    """
    u = tf.sort(x, direction="DESCENDING")
    indrange = tf.range(u.shape[0]) + 1
    cumsum = tf.cast(tf.cumsum(u), tf.float32)
    ro = tf.reduce_max(indrange[u + 1. / tf.cast(indrange, tf.float32) * (1 - cumsum) > 0])
    # ro = tf.cast(ro)
    lm = 1 / tf.cast(ro, tf.float32) * (1 - cumsum[ro - 1])
    return tf.nn.relu(x + lm)


def unit_ball_proj(x):
    """
    Function of projection on the unit ball in R^n w.r.t. Euclidean norm
    """
    if tf.linalg.norm(x) > 1:
        x = x / tf.linalg.norm(x)
    return x


def l1_ball_proj(x, R):
    """
        Function of projection on the unit ball in R^n w.r.t. Euclidean norm
    """
    u = tf.cast(tf.sort(x), tf.float32)
    indrange = tf.range(u.shape[0]) + 1
    cumsum = tf.cumsum(u)
    ro = tf.reduce_max(indrange[u - 1. / tf.cast(indrange, tf.float32) * (cumsum - R) > 0])
    lm = tf.cast(1 / ro, tf.float32) * (cumsum[ro - 1] - R)
    lm = tf.cast(lm, tf.float32)
    res = tf.where(x > lm, x - lm, tf.zeros(x.shape, dtype=tf.float32))
    res = tf.where(x < lm, x + lm, res)
    return tf.cast(res, tf.float32)


def bad_func(x, L=2):
    """
        x: tf.Variable or tf.Tensor, shape=(n,), n >= 3.
    """
    z = x[0] ** 2
    z += tf.reduce_sum((x[1:] - x[:-1]) ** 2)
    z += x[-1] ** 2
    z *= 1 / 2.
    z -= x[0]
    z *= L / 4.
    return z


def run_astm(func, lm0, L, prox, R, max_iter=100, eps=1e-7):
    """
    Run Adaptive Similar Triangle Method.
    Parameters:
        func: callable. Function to optimize. Takes d-dimensional np.array as input and outputs a scalar
        grad_func: callable. Gradient of func. Takes d-dimensional np.array as input and outputs a d-dimensional np.array
        lm_0: initial point lambda_0
        L: initial guess L_0
        prox: proximal function d(lambda), e.g. L-1 regularizer. Must be 1-strongly convex
        grad_prox: gradient of prox
        R:
        max_iter: maximum number of iterations
        eps: float, tolerance of the algorithm
    """
    alpha = 0
    c = 0
    eta = tf.Variable(lm0)
    zeta = tf.Variable(lm0)
    logging.warn("L = {}".format(L))
    loss_history = []
    for k in tqdm.tqdm(range(max_iter)):
        # Step 3: Caclculate M_k = L_k / 2
        m = L / 2
        while True:
            # Step 5: calculate \alpha_{k+1} and C_{k+1}
            m *= 2
            alpha = np.max(solve_quadratic(-m, 1, c))

            c_new = c + alpha

            # Step 6: calculate \lambda_{k+1}
            ll = tf.Variable((alpha * zeta + c * eta) / c_new)

            # Step 7: calculate \zeta_{k+1}
            with tf.GradientTape() as tape:
                tape.watch(zeta)
                prox_zeta = prox(zeta)
            grad_prox_zeta = tape.gradient(prox_zeta, zeta)

            with tf.GradientTape() as tape:
                tape.watch(ll)
                func_ll = func(ll)
            grad_func_ll = tape.gradient(func_ll, ll)
            optimizee = (
                lambda lm: prox(lm)
                - prox(zeta)
                - tf.tensordot(grad_prox_zeta, (lm - zeta), 1)
                + alpha * (func(ll) + tf.tensordot(grad_func_ll, (lm - ll), 1))
            )
            zeta_t = minimize(optimizee, tf.Variable(ll + np.random.rand(*ll.shape)))

            # Step 8: calculate \eta_{k+1}
            eta_t = (alpha * zeta_t + c * eta) / c_new
            # Check the break condition
            if (
                func(eta_t)
                <= func(ll)
                + tf.tensordot(grad_func_ll, (eta_t - ll), 1)
                + m / 2 * tf.linalg.norm(eta_t - ll) ** 2
            ):
                break

        c = c_new
        L = m / 2
        zeta = zeta_t
        eta = eta_t
        # Option 2
        if R ** 2 / c <= eps:
            break

        # Option 2
        # if func(eta) -
        # print(eta_t)
        loss_history.append(func(eta))
    return eta, loss_history


def run_adaptive_gm():
    """
    Run Adaptive Gradient method.
    """


def run_fw(func, lm0, proj=None, max_iter=100, eps=1e-7):
    """
    Run Frank-Wolfe algorithm.
    Parameters:
        func: callable. Function to optimize. Takes d-dimensional np.array as input and outputs a scalar
        lm0: tf.Variable, initial point
    """
    loss_history = []
    for t in tqdm.tqdm(range(max_iter)):
        eta = 2 / (2 + t)
        with tf.GradientTape() as tape:
            tape.watch(lm0)
            loss = func(lm0)
        loss_history.append(loss)
        grad = tf.cast(tape.gradient(loss, lm0), tf.float32)
        optimizee = lambda x: tf.tensordot(grad, x, 1)
        st = minimize(optimizee, tf.Variable(proj(tf.cast(grad + tf.random.normal(shape=grad.shape), tf.float32))), proj)
        st = tf.cast(st, tf.float64)
        lm0 = (1 - eta) * lm0 + eta * st
        print(lm0, loss)
    return lm0, loss_history


def run_amd(func, g, x0, prox, proj, max_iter=100, R=1000, eps=1e-6):
    """
    Run Adaptive Mirror Descent for non-smooth objective.
    Parameters:
        func: callable. Objective to minimize.
        x0: tensor. shape=(n,) n >= 3. Initial point of optimization.
        prox: callable. prox-function
        proj: callable. Function of projection onto some set
        max_iter: integer. Maximum number of iterations.
        R: float. Such constant that prox(x_*) <= R
        eps: float. Accuracy.
    """
    step_history = []
    h_history = []
    cumulative_m = 0
    loss_history = []
    for k in tqdm.tqdm(range(max_iter)):
        # compute gradients of f and g
        with tf.GradientTape() as tape:
            tape.watch(x0)
            loss = func(x0)
        grad = tape.gradient(loss, x0)
        with tf.GradientTape() as tape:
            tape.watch(x0)
            constraint = g(x0)
        grad_g = tape.gradient(constraint, x0)
        loss_history.append(loss)
        if g(x0) <= eps:
            # mirr[x](p) = argmin_u {<p, u> + d(u) - <\nabla d(x), u>}
            # mirr[x_k](hk * grad) = argmin_{u} {<hk * grad, u> + d(u) - <\nabla d(x_k), u>}
            M = tf.linalg.norm(grad)
            hk = eps / (M ** 2)
            # compute prox-gradient w.r.t. x0
            with tf.GradientTape() as tape:
                prox_xk = prox(x0)
            grad_prox_xk = tape.gradient(prox_xk, x0)
            optimizee = (
                lambda u: tf.tensordot(hk * grad, u, 1)
                + prox(u)
                - tf.tensordot(grad_prox_xk, u, 1)
            )
            x0 = minimize(optimizee, tf.Variable(x0 + np.random.rand(*x0.shape)), proj)
            step_history.append(x0)
            h_history.append(hk)
        else:
            M = tf.linalg.norm(grad_g)
            hk = eps / (M ** 2)
            with tf.GradientTape() as tape:
                prox_xk = prox(x0)
            grad_prox_xk = tape.gradient(prox_xk, x0)
            optimizee = (
                lambda u: tf.tensordot(hk * grad_g, u, 1)
                + prox(u)
                - tf.tensordot(grad_prox_xk, u, 1)
            )
            x0 = minimize(optimizee, tf.Variable(x0 + np.random.rand(*x0.shape)), proj)
        cumulative_m += 1 / M ** 2
        # check the termination condition
        if cumulative_m >= 2 * R ** 2 / eps ** 2:
            break
    return sum(h * x for h, x in zip(h_history, step_history)) / sum(h_history), loss_history


if __name__ == "__main__":
    pass