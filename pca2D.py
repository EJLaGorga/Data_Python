import numpy as np
import matplotlib.pyplot as plt

def plot_principle_components_2D(X):
    """ Given a two dimensional array of numbers, X, where each row represents
    an observation, plot the observations in R^2, and overlay the principle
    directions.

    :param X: A 2D numpy array
    """

    def plotvec(v, **kwargs):
        """ Plot a vector in the plane.
        :param v: a shape (2,) or (2,1) numpy array
        """
        plt.plot([0, v[0]], [0, v[1]], **kwargs)

    cov = np.cov(X.T, ddof=1)
    [w,v] = np.linalg.eig(cov)
    vec1 = w[0]*v[:,0]
    vec2 = w[1]*v[:,1]
    v = np.concatenate([vec1, vec2], axis=1)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plotvec(vec1, color='g')
    plotvec(vec2, color='r')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

def plot_dim_reduced_2D(X):
    """ Plots each point in X projected on to the principle component.

    :param X: A 2D numpy array
    """

    def plotvec(v, **kwargs):
        """ Plot a vector in the plane.
        :param v: a shape (2,) or (2,1) numpy array
        """
        plt.plot([0, v[0]], [0, v[1]], **kwargs)

    cov = np.cov(X.T, ddof=1)
    w,v = np.linalg.eig(cov)
    i = w.argsort()[::-1]
    w = w[i]
    v = v[:,i]

    op = np.outer(v[:,0],v[:,0])
    p = np.dot(X,op)

    plt.figure()
    plt.scatter(p[:, 0], p[:, 1])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
	

if __name__ == "__main__":
    # Below is boilerplate code to get your up and running. The particular
    # value of X will be changed when your code is exercised.
    X = np.random.multivariate_normal(mean=(0, 0), cov=((1, 2), (0.5, 3)),
                                      size=100)
    plot_principle_components_2D(X)
    plot_dim_reduced_2D(X)



