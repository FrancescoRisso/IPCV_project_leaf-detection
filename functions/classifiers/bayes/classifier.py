from typing import Any

import json

from sklearn.preprocessing import KBinsDiscretizer  # type: ignore


def BAYES_classify(new_data: dict[str, Any]) -> dict[str, float]:
    """
    Loads the Bayesian classifier model from file, and uses it to perform
    a classification task on a new data (described as feature values)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - new_data: the features of the new image to be classified

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A dictionary that associates each plant to the probability estimated
    for the image to be that specific plant
    """

    with open("./classification_models_data/bayes.json", "r") as f:
        model = json.load(f)

    features = [f for f in model["discretization"].keys()]
    leaves = [l for l in model["P(C)"].keys()]

    res: dict[str, float] = dict.fromkeys(leaves, 1.0)

    for feature in features:
        val = __discretize_feature_val(
            model["discretization"][feature], new_data[feature]
        )

        for leaf in leaves:
            P_C = model["P(C)"][leaf]
            P_X_given_C = model["P(X|C)"][feature][leaf][val]
            res[leaf] *= P_C * P_X_given_C

        sum = 0.0
        for leaf in leaves:
            sum += res[leaf]

        for leaf in leaves:
            res[leaf] /= sum

    return res


def __discretize_feature_val(model: dict[str, Any], value: Any) -> int:
    """
    Given the new data value, discretizes it in the same way the feature
    was discretize at the time of dataset evaluation

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - model: the json data about this feature
        (```json["discretization"][feature]```)
    - value: the original value from the new data

    ---------------------------------------------------------------------
    OUTPUT
    ------
    Value discretized according to model, as an integer class ID
    """
    if model["bin_edges"] is None:
        return 1 if value else 0

    discretizer = KBinsDiscretizer(
        n_bins=model["num_bins"], strategy="quantile", encode="ordinal"
    )
    discretizer.bin_edges_ = [model["bin_edges"]]

    return int(discretizer.transform([[value]])[0])
