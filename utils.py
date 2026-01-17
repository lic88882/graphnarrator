from typing import Iterable, List


def safe_zip(*iterables: List[Iterable]) -> Iterable:
    """
    Zip function that ensures all iterables are of the same length.

    Args:
        *iterables: A list of iterables to zip together

    Returns:
        Iterable: An iterable of tuples containing the elements from each input iterable
    """
    # Ensure all iterables are the same length
    lengths = [len(iterable) for iterable in iterables]
    if len(set(lengths)) > 1:
        raise ValueError("safe_zip requires all iterables to be of the same length")
    return zip(*iterables)
