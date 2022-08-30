import numpy as np
from pymsm.utils import stepfunc


def test_stepfunc():
    xs = [1, 2, 3, 5]
    ys = [1, 2, 3, 4]

    xnew = np.arange(0, 6, 0.01)
    ynew = stepfunc(xs, ys)(xnew)

    np.testing.assert_equal(ynew[np.where(xnew == 0.5)[0]], 0)
    np.testing.assert_equal(ynew[np.where(xnew == 1.0)[0]], 1)

    np.testing.assert_equal(ynew[np.where(xnew == 1.5)[0]], 1)
    np.testing.assert_equal(ynew[np.where(xnew == 2.0)[0]], 2)

    np.testing.assert_equal(ynew[np.where(xnew == 2.5)[0]], 2)
    np.testing.assert_equal(ynew[np.where(xnew == 3.0)[0]], 3)

    np.testing.assert_equal(ynew[np.where(xnew == 3.5)[0]], 3)
    np.testing.assert_equal(ynew[np.where(xnew == 4.0)[0]], 3)
    np.testing.assert_equal(ynew[np.where(xnew == 5.0)[0]], 4)
