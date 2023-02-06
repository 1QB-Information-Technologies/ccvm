"""Utility functions and classes."""


def imean(iterator: iter) -> float:
    """Take the mean of an iterator.

    If taking the mean of an iterable that is not an iterator, you might be
    better off using numpy.mean.

    Args:
        iterator (iter): Iterator to take the mean of.

    Returns:
        float: The mean of all elements in the iterator.
    """
    # It's necessary to spell out how to do a sum since numpy.mean doesn't support using
    # iterators
    sum_ = 0.0
    n = 0
    for el in iterator:
        sum_ += el
        n += 1

    # sum_ is float so the division will not be truncated
    return sum_ / n


def ivariance(iterator: iter) -> float:
    """Return the variance of an iterator.

    Args:
        iterator (iter): Iterator to take the variance of.

    Returns:
        float: The variance of all elements in the iterator.
    """
    sum_ = 0.0
    sum_2 = 0.0
    n = 0
    for e in iterator:
        sum_ += e
        sum_2 += e**2
        n += 1

    return (sum_2 - (sum_**2 / n)) / n
