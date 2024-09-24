import os
import cv2

from functions.features import ImageFeatures


def update_dataset() -> None:
    leaves = os.listdir("./dataset")

    for leaf in leaves:
        files_list = os.listdir(f"./dataset/{leaf}")
        print(f'Updating dataset for plant "{leaf}"...')

        for img_file_name in files_list:
            # Do not process json files
            if img_file_name.endswith(".json"):
                continue

            img_path = f"./dataset/{leaf}/{img_file_name}"
            json_path = f"./dataset/{leaf}/{os.path.splitext(img_file_name)[0]}.json"

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

        # TODO compute plant summary and percentages

    print(f"Dataset update complete!")


if __name__ == "__main__":
    update_dataset()
