import os
import cv2

from functions.features import ImageFeatures
from functions.classifiers.bayes.summarize_dataset import BAYES_summarize_dataset

import threading
import json


def update_dataset() -> None:
    print(f"Updating dataset...")

    leaves = os.listdir("./dataset/images")
    threads = []

    for leaf in leaves:
        threads.append(threading.Thread(target=process_plant, args=(leaf,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("\nDataset update complete!")
    print("Updating bayes model...")
    BAYES_summarize_dataset()
    print("Bayes model update complete!")


def process_plant(leaf: str) -> None:
    files_list = os.listdir(f"./dataset/images/{leaf}")

    # If the descriptions folder does not exist, create it
    if not os.path.exists(f"./dataset/descriptions/{leaf}"):
        os.makedirs(f"./dataset/descriptions/{leaf}")

    all_leaves_list = []

    for img_file_name in files_list:
        img_path = f"./dataset/images/{leaf}/{img_file_name}"
        json_path = (
            f"./dataset/descriptions/{leaf}/{os.path.splitext(img_file_name)[0]}.json"
        )

        img = cv2.imread(img_path)
        img_features = ImageFeatures(img)

        if os.path.exists(json_path):
            json_last_modify = os.path.getmtime(json_path)
            img_last_modify = os.path.getmtime(img_path)

            if img_last_modify < json_last_modify:
                # Load json data only if the json exists and the image has not changed since its computation
                img_features.load_details_from_file(json_path)

        # If there were updates, update the file
        img_features.store_to_file(json_path)
        all_leaves_list.append(img_features.get_features())

    all_leaves_data = {}
    for feature in all_leaves_list[0].keys():
        all_leaves_data[feature] = [leaf[feature] for leaf in all_leaves_list]

    with open(f"./dataset/plant_recaps/{leaf}.json", "w") as f:
        json.dump(all_leaves_data, f)

    print(f'Dataset for plant "{leaf}" is now updated.')


if __name__ == "__main__":
    update_dataset()
