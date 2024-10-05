from typing import TypeVar, List


T = TypeVar("T")
type tuple_of_11[T] = tuple[T, T, T, T, T, T, T, T, T, T, T]


def to_tuple_of_11(array: List[T]) -> tuple_of_11[T]:
    """
    Equivalent of python's tuple() function, to convert an array of 11
    items into a tuple_of_11

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - array: the array to be converted

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The tuple, if the array had 11 elements.
    Otherwhise, an error is raised
    """
    if len(array) != 11:
        raise ValueError("Array must have exactly 11 elements")
    return (
        array[0],
        array[1],
        array[2],
        array[3],
        array[4],
        array[5],
        array[6],
        array[7],
        array[8],
        array[9],
        array[10],
    )

def tuple_of_11_to_python_tuple(orig: tuple_of_11[T]) -> tuple[T, ...]:
    """
    Converts the tuple to a JSON object

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - orig: the tuple_of_11 to be converted
     
    ---------------------------------------------------------------------
    OUTPUT
    ------
    The tuple as JSON object (a python tuple)
    """
    return tuple(orig)
