import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def sir_1(X, y, H, K):
    """Sliced Inverse Regression : Method 1

        Parameters
        ----------
        X : float, array of shape [n_samples, dimension]
            The generated samples.

        y : int, array of shape [n_samples, ]
            Labels of each generated sample.

        H : number of slices

        K : return K estimated e.d.r directions

        Returns
        -------
        edr_est : estimated e.d.r directions
        """
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X-np.mean(X, axis=0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y)+h*width <= y, y < np.min(y)+(h+1)*width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        mh = np.mean(Z[h_index, :], axis=0)
        V_hat += ph_hat * np.matmul(mh[:, np.newaxis], mh[np.newaxis, :])

    # 特征值和特征向量（列向量）
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est = np.matmul(r, K_largest_eigenvectors)

    for k in range(K):
        scatter_plot(x=np.matmul(X, edr_est[:, k]), y=y, x_name=str(k+1) + "-th estimated direction", y_name="Y", title=None,
                     file_name="sir1_" + str(k+1) + "-th_est_direction.png")

    return edr_est


def sir_2(X, y, H, K):
    """Sliced Inverse Regression : Method 2

        Parameters
        ----------
        X : float, array of shape [n_samples, dimension]
            The generated samples.

        y : int, array of shape [n_samples, ]
            Labels of each generated sample.

        H : number of slices

        K : return K estimated e.d.r directions

        Returns
        -------
        edr_est : estimated e.d.r directions
        """
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X - np.mean(X, axis=0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y) + h * width <= y, y < np.min(y) + (h + 1) * width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        V_hat += np.matmul(Z[h_index, :].T, Z[h_index, :]) / ph_hat

    V_hat = V_hat/(H*X.shape[0])
    # 特征值和特征向量（列向量）
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(1-eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est = np.matmul(r, K_largest_eigenvectors)

    for k in range(K):
        scatter_plot(x=np.matmul(X, edr_est[:, k]), y=y, x_name=str(k + 1) + "-th estimated direction", y_name="Y", title=None,
                     file_name="sir2_" + str(k + 1) + "-th_est_direction.png")

    return edr_est


def save(X, y, H, K):
    """Sliced Average Variance Estimates

        Parameters
        ----------
        X : float, array of shape [n_samples, dimension]
            The generated samples.

        y : int, array of shape [n_samples, ]
            Labels of each generated sample.

        H : number of slices

        K : return K estimated e.d.r directions

        Returns
        -------
        edr_est : estimated e.d.r directions
        """
    cov_x = np.cov(X, rowvar=False)
    u = np.linalg.inv(cov_x)
    r = np.array(sqrtm(u))
    Z = np.matmul(X - np.mean(X, axis=0), r)

    width = (np.max(y) - np.min(y)) / H

    V_hat = np.zeros([X.shape[1], X.shape[1]])
    for h in range(H):
        h_index = np.logical_and(np.min(y) + h * width <= y, y < np.min(y) + (h + 1) * width)
        ph_hat = np.mean(h_index)
        if ph_hat == 0:
            continue
        temp = np.eye(X.shape[1]) - np.matmul(Z[h_index, :].T, Z[h_index, :]) / (ph_hat*X.shape[0])
        V_hat += np.matmul(temp, temp)

    # 特征值和特征向量（列向量）
    eigenvalues, eigenvectors = np.linalg.eig(V_hat)
    K_index = np.argpartition(np.abs(eigenvalues), X.shape[1]-K) >= X.shape[1]-K
    K_largest_eigenvectors = eigenvectors[:, K_index]
    edr_est = np.matmul(r, K_largest_eigenvectors)

    for k in range(K):
        scatter_plot(x=np.matmul(X, edr_est[:, k]), y=y, x_name=str(k + 1) + "-th estimated direction", y_name="Y", title=None,
                     file_name="save_" + str(k + 1) + "-th_est_direction.png")

    return edr_est


def scatter_plot(x, y, x_name, y_name, title, file_name, dpi=200):
    # 绘制二维图片
    origin_dpi = plt.rcParams['figure.dpi']
    plt.rcParams['figure.dpi'] = dpi
    plt.scatter(x, y, s=10, color="b", alpha=0.5)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    plt.savefig(file_name)
    plt.close()
    plt.rcParams['figure.dpi'] = origin_dpi


# # example 1 : y = x_1 + x_2 + x_3 + x_4 + 0x_5 + epsilon
# n = 100
# H = 5
# K = 1
# beta = np.array([1, 1, 1, 1, 0])
# print('beta : ', beta)
# X = np.random.normal(0, 1, [n, 5])
# epsilon = np.random.normal(0, 1, n)
# y = np.matmul(X, beta) + epsilon
#
# print('SIR Method 1, beta_hat : ')
# print(sir_1(X, y, H=H, K=K))
#
# print('SIR Method 2, beta_hat : ')
# print(sir_2(X, y, H=H, K=K))
#
# print('SAVE, beta_hat : ')
# print(save(X, y, H=H, K=K))


# # example 2 : y = x_1 * ( x_1 + x_2 + 1 ) + sigma * epsilon
# n = 400
# H = 5
# K = 2
# p = 10  # number of variables
# sigma = 0.5
# beta1 = np.concatenate([[1], np.zeros(p-1)])
# beta2 = np.concatenate([[0, 1], np.zeros(p-2)])
# print('beta1 : ', beta1)
# print('beta2 : ', beta2)
# X = np.random.normal(0, 1, [n, p])
# epsilon = np.random.normal(0, sigma, n)
# y = np.matmul(X, beta1) * (np.matmul(X, beta1) + np.matmul(X, beta2) + 1.) + epsilon
#
# print('Method 1, beta_hat : ')
# print(sir_1(X, y, H=H, K=K))
#
# print('Method 2, beta_hat : ')
# print(sir_2(X, y, H=H, K=K))


# # example 3 : y = x_1 / ( 0.5 + ( x_2 + 1.5 )^2 ) + sigma * epsilon
# n = 400
# H = 5
# K = 2
# p = 10  # number of variables
# sigma = 0.5
# beta1 = np.concatenate([[1], np.zeros(p-1)])
# beta2 = np.concatenate([[0, 1], np.zeros(p-2)])
# print('beta1 : ', beta1)
# print('beta2 : ', beta2)
# X = np.random.normal(0, 1, [n, p])
# epsilon = np.random.normal(0, sigma, n)
# y = np.matmul(X, beta1) / (0.5 + np.power(np.matmul(X, beta2) + 1.5, 2)) + epsilon
#
# print('Method 1, beta_hat : ')
# print(sir_1(X, y, H=H, K=K))
#
# print('Method 2, beta_hat : ')
# print(sir_2(X, y, H=H, K=K))


# example 4 : y = x_1^2 + epsilon
n = 400
H = 5
K = 1
v = 0.5
beta = np.array([1, 0])
print('beta : ', beta)
X = np.zeros([n, 2])
X[:, 0] = np.random.normal(0, 1, n)
X[:, 1] = X[:, 0] * X[:, 0] + np.random.normal(0, v, n)
epsilon = np.random.normal(0, 1, n)
y = X[:, 0] * X[:, 0] + epsilon

print('Method 1, beta_hat : ')
print(sir_1(X, y, H=H, K=K))

print('Method 2, beta_hat : ')
print(sir_2(X, y, H=H, K=K))

print('SAVE, beta_hat : ')
print(save(X, y, H=H, K=K))
