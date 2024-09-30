import os
import cv2

from functions.lengths.paper_roi import find_roi_boundaries, roi_boundaries_as_rect

from functions.utils.image import draw_rectangle


def main() -> None:
    leaves = os.listdir("./dataset")
    for leaf in leaves:
        imgs = os.listdir(f"./dataset/{leaf}")
        for img_name in imgs:
            img = cv2.imread(f"./dataset/{leaf}/{img_name}")
            print(f"./dataset/{leaf}/{img_name}")

            roi_tuple = find_roi_boundaries(img)
            roi = roi_boundaries_as_rect(roi_tuple)
            img = draw_rectangle(img, roi, (0, 0, 255), 5)

            cv2.imwrite(f"test/{leaf}.jpg", img)

            break
            # return


if __name__ == "__main__":
    main()
