import os
import json


def clear_dataset_feature() -> None:
    for leaf in os.listdir("./dataset/descriptions"):
        for file_name in os.listdir(f"./dataset/descriptions/{leaf}"):
            with open(f"./dataset/descriptions/{leaf}/{file_name}", "r") as file:
                cache = json.load(file)

            # Delete here what you want to delete
            # example: del cache["features"]["height"]
            
            # Done

            with open(f"./dataset/descriptions/{leaf}/{file_name}", "w") as file:
                json.dump(cache, file)

    print("Done")
