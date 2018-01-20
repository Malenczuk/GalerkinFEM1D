import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def p1(h, k):
    result = -1 / h
    x_k = k * h
    if x_k <= 1:
        result += h / 6
    elif x_k - h >= 1:
        result += 2 * h / 6
    else:
        result += h / 4
    return result


def p2(h, k):
    result = 2 / h
    x_k = k * h
    if x_k == 1:
        result += h
    elif x_k + h <= 1:
        result += 2 * h / 3
    elif x_k - h >= 1:
        result += 4 * h / 3
    elif (x_k - h < 1) and (x_k > 1):
        result += 31 * h / 24
    elif (x_k < 1) and (x_k + h > 1):
        result += 17 * h / 24
    return result


def p3(h, k):
    result = -1 / h
    x_k = k * h
    if x_k + h <= 1:
        result += h / 6
    elif x_k >= 1:
        result += 2 * h / 6
    else:
        result += h / 4
    return result


def fem(g0, u2, f, n):
    h = 2.0 / n
    matrix = np.zeros((n, n))

    for k in range(n):
        matrix[k][k] = p2(h, k)
        if k > 0:
            matrix[k - 1][k] = p3(h, k - 1)
            matrix[k][k - 1] = p1(h, k)

    matrix[0][0] = (1 / h) - (h / 3)
    vector = np.zeros((n, 1))

    for k in range(n):
        vector[k] = 4 / 3 * f(h * k) * h

    vector[0] = (f(0) + 2 * f(h / 2)) * h / 6 - g0
    vector[n - 1] -= u2 * ((-1 / h) + (2 * h / 6))

    result = la.solve(matrix, vector)

    points = np.linspace(0.0, 2.0, n + 1)
    values = np.zeros(n + 1)

    for k in range(n):
        values[k] = result[k]

    values[n] = u2

    plt.plot(points, values)
    plt.show()


def main():
    fem(0, 0, lambda x: x, 100)


if __name__ == "__main__":
    main()
