import os
import json


def clear_dataset_feature(feature: str) -> None:
    for leaf in os.listdir("./dataset/descriptions"):
        for file_name in os.listdir(f"./dataset/descriptions/{leaf}"):
            with open(f"./dataset/descriptions/{leaf}/{file_name}", "r") as file:
                cache = json.load(file)

            del cache["features"][feature]

            with open(f"./dataset/descriptions/{leaf}/{file_name}", "w") as file:
                json.dump(cache, file)

    print("Done")
