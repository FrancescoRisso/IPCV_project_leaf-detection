MIN_LEAF_HUE = 0
MAX_LEAF_HUE = 80
MIN_LEAF_SAT = 110
MAX_LEAF_VAL = 150


def is_px_leaf(px: tuple[int, int, int]) -> bool:
    """
    Looking at the colors, tells if a pixel can be part of a leaf or not

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - px: the pixel to analyse, described as HSV tuple

    ---------------------------------------------------------------------
    OUTPUT
    ------
    Whether the pixel can belong to a leaf or not
    """
    
    hue, sat, val = px

    if (
        hue >= MIN_LEAF_HUE
        and hue <= MAX_LEAF_HUE
        and sat >= MIN_LEAF_SAT
        and val <= MAX_LEAF_VAL
    ):
        return True
    
    return False
