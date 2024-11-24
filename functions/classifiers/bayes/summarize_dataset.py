import json

from math import floor, log2, log10, sqrt

import os

from sklearn.preprocessing import KBinsDiscretizer  # type: ignore


from typing import Any


def BAYES_summarize_dataset() -> None:
    """
    Computes all the values required for the bayesian classifier to work

    It does:
    - load the plant recap for all different plants
    - choose how to discretize the features
    - discretize the features
    - compute all the probabilities required for the classification
    - store them to a file
    """

    # Load data
    labels, data = __load_all_data()

    to_store: dict[str, Any] = {"discretization": {}, "P(X|C)": {}}
    data_discrete = {}

    # Discretize all features and store the discretization parameters
    for feature in data.keys():
        num_bins, bin_edges, discretized = __discretize_data(data[feature], labels)
        data_discrete[feature] = discretized

        to_store["discretization"][feature] = {}
        to_store["discretization"][feature]["num_bins"] = num_bins
        to_store["discretization"][feature]["bin_edges"] = bin_edges

    # Compute and store P(C) for each plant C
    P_C = __compute_leaf_percentages(labels)
    to_store["P(C)"] = P_C

    # Compute and store P(X|C) for each plant C for each feature X
    for feature in data.keys():
        P_X_given_C = __compute_feature_given_leaf_percentages(
            labels,
            data_discrete[feature],
            to_store["discretization"][feature]["num_bins"],
        )
        to_store["P(X|C)"][feature] = P_X_given_C

    # Store all values to the classification model data folder
    with open("./classification_models_data/bayes.json", "w") as f:
        json.dump(to_store, f)

    # Store all data as csv file for computing correlation matrix
    with open("./dataset/alldata.csv", "w") as f:
        f.write(",".join(data.keys()))
        f.write("\n")
        for i in range(len(labels)):
            f.write(
                ",".join([str(data_discrete[feature][i]) for feature in data.keys()])
            )
            f.write("\n")


def __load_all_data() -> tuple[list[str], dict[str, Any]]:
    """
    Loads all the data from all the plants, putting all the plants' data
    in the same array, divided by feature.

    It also creates a list of corresponding labels

    ---------------------------------------------------------------------
    OUTPUT
    ------
    - a list of labels, ordered in the same way as the data
    - a dict with the features as keys, and the list of values for that
        feature as value

    ---------------------------------------------------------------------
    EXAMPLE
    -------

    Suppose to have features F1 and F2, and plants p1, p2, p3:
    - p1 has 3 leaves, with values (F1, F2) = ```[(10,10), (11,11), (4,4)]```
    - p2 has 2 leaves, with values (F1, F2) = ```[(4,5), (5,4)]```
    - p3 has 1 leaf, with values (F1, F2) = ```[(1,3)]```

    The result will be:
    ```
    (
        ["p1", "p1", "p1", "p2", "p2", "p3"],
        {
            "F1": [10, 11, 4, 4, 5, 1],
            "F2": [10, 11, 4, 5, 4, 3]
        }
    )
    ```

    """

    data: dict[str, Any] = {}
    plants: list[str] = []

    for plant in os.listdir(f"./dataset/images"):
        with open(f"./dataset/plant_recaps/{plant}.json", "r") as plant_recap_file:
            plant_data = json.load(plant_recap_file)

        if len(data.keys()) == 0:
            for key in plant_data:
                data[key] = []

        for key in data.keys():
            data[key] = [*data[key], *plant_data[key]]
            num_imgs = len(plant_data[key])

        plants = [*plants, *[plant for _ in range(num_imgs)]]

    return plants, data


def __discretize_data(
    data: list[Any], plants: list[str]
) -> tuple[int, list[float] | None, list[int]]:
    """
    Given data for a feature, finds the best way to discretize it, and
    then executes the discretization.

    If the attribute is boolean, False = 0 and True = 1

    For float values, the discretization is based on quantiles, and the
    number of bins is selected as the one with greatest entropy among:
    - ```10```
    - ```max(1, floor(2 * log10(len(data))))```
    - ```floor(1 + log2(len(data)))```
    - ```floor(sqrt(len(data)))```

    These values are some common choices, as described in
    [this paper](www.litrp.cl/cwpr2013/papers/jcc2013_submission_192.pdf)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - data: the list of data values for a given feature
    - plant: the list of plant names associated to data

    ---------------------------------------------------------------------
    OUTPUT
    ------
    - the selected number of bins
    - the selected bins (discretizer.bin_edges_), or None if the feature
        was boolean
    - the discretized data

    """

    if type(data[0]) == bool:
        return (2, None, [1 if val else 0 for val in data])

    data = [[val] for val in data]

    plants_count_empty = json.dumps(dict.fromkeys([plant for plant in set(plants)], 0))

    # Store the score and settings for the best bin number
    max_score = 0.0
    max_score_bins = 0
    max_score_edges = []
    max_score_discretization = []

    # Compute the options for the number of bins
    num_bins_options = [
        10,
        max(1, floor(2 * log10(len(data)))),
        floor(1 + log2(len(data))),
        floor(sqrt(len(data))),
    ]

    for num_bins in num_bins_options:
        # Create discretizer
        discretizer = KBinsDiscretizer(
            n_bins=num_bins, strategy="quantile", encode="ordinal"
        )

        # Discretize features
        binned = discretizer.fit_transform(data)

        # Count entries for each (bin, plant) and bin
        count = [json.loads(plants_count_empty) for _ in range(num_bins)]
        elements_per_bin = [0 for _ in range(num_bins)]
        for plant, bin_num in zip(plants, binned):
            bin_num = int(bin_num)
            count[bin_num][plant] += 1
            elements_per_bin[bin_num] += 1

        # Compute entropy
        entropy = 0.0
        splitinfo = 0.0
        for bin_num in range(num_bins):
            for leaf_count in count[bin_num].values():
                p = (
                    float(leaf_count) / elements_per_bin[bin_num]
                    if elements_per_bin[bin_num] != 0
                    else 0.0
                )
                entropy_here = -p * log2(p) if p != 0 else 0.0
                entropy += entropy_here

            p = float(elements_per_bin[bin_num]) / len(data)
            entropy_here = -p * log2(p) if p != 0 else 0.0
            splitinfo += entropy_here

        # Compute score and update max score settings if needed
        score = entropy / splitinfo
        if score > max_score:
            max_score = score
            max_score_bins = num_bins
            max_score_edges = discretizer.bin_edges_
            max_score_discretization = binned

    # Return the values, while transforming the discretization from float to int
    return (
        max_score_bins,
        list(max_score_edges[0]),
        [int(val) for val in max_score_discretization],
    )


def __compute_leaf_percentages(plants: list[str]) -> dict[str, float]:
    """
    Given the set of labels of the dataset, computes the probability of
    a random image to have each class (to be of each given leaf).

    In Bayes model, this is P(C)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - plants: a list with the label names, each inserted once per each
        image of that leaf present in the dataset

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A dict that associates each plant to the frequency of that plant in
    the dataset
    """

    count = dict.fromkeys([plant for plant in sorted(set(plants))], 0)

    for plant in plants:
        count[plant] += 1

    res = {}

    for plant in count.keys():
        res[plant] = float(count[plant]) / len(plants)

    return res


def __compute_feature_given_leaf_percentages(
    plants: list[str], data: list[int], num_bins: int
) -> dict[str, list[float]]:
    """
    Given the discretized data for a feature, computes the probability of
    a random leaf of a specific class to have each discrete value, for
    each class.

    In Bayes model, this is P(X|C)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - plants: a list with the label names, each inserted once per each
        image of that leaf present in the dataset
    - data: a list with the discretized data values, in direct
        correspondance with plants
    - num_bins: the number of bins used to discretize the specific
        feature

    ---------------------------------------------------------------------
    OUTPUT
    ------
    A dict taht associates each plant (C) to an array. This array
    associates each discrete value (the index, X) to the probability
    P(X|C)
    """

    # Laplace smoothing: each data point counts as 3, and each bin has 1 extra count

    count = dict.fromkeys([plant for plant in sorted(set(plants))], num_bins)

    count_per_bin: dict[str, list[int]] = dict.fromkeys(
        [plant for plant in set(plants)], []
    )
    for plant in count_per_bin.keys():
        count_per_bin[plant] = [1 for _ in range(num_bins)]

    for plant, bin_num in zip(plants, data):
        count[plant] += 3
        count_per_bin[plant][bin_num] += 3

    res: dict[str, list[float]] = {}

    for plant in count.keys():
        res[plant] = []

        for bin_num in range(num_bins):
            perc = float(count_per_bin[plant][bin_num]) / count[plant]
            res[plant].append(perc)

    return res
