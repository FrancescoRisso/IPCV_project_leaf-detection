def print_classification_result(perc: dict[str, float], verbose: bool) -> None:
    """
    Prints the result of a classification task in a tidy way

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - perc: a dict in the format ```{"plant": percentage}```
    """

    max = -1.0
    argmax = ""
    sum = 0.0

    for leaf, val in perc.items():
        if val > max:
            max = val
            argmax = leaf

        sum += val

    print(f'Classified as "{argmax}" with confidence {(max/sum*100):2.4f}%')
    if verbose:
        print("Full classification result:")
        for leaf, val in reversed(sorted(perc.items(), key=lambda x: x[1])):
            print(f"- {(val/sum*100):8.4f}% --> {leaf}")
