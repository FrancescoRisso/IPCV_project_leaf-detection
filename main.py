import os
import cv2

from functions.utils.rectangle import Rectangle
from functions.utils.segment import Segment
from functions.utils.image import draw_rectangle

from functions.lengths.leaf_width import get_leaf_widths, get_leaf_roi
from functions.lengths.leaf_height import find_leaf_height


pos = {
    "acero rubrum": 1,
    "liquidambar": 4,
    "gaggia": 2,
    "mirtillo": 3,
    "salvia": 5,
    "ulivo": 6,
}

rois: dict[str, Rectangle] = {
    "acero rubrum": Rectangle(Segment(296, 4744), Segment(864, 6752)),
    "liquidambar": Rectangle(Segment(0, 4776), Segment(605, 6846)),
    "gaggia": Rectangle(Segment(143, 4796), Segment(720, 6776)),
    "mirtillo": Rectangle(Segment(496, 4312), Segment(1651, 6323)),
    "salvia": Rectangle(Segment(187, 4697), Segment(770, 6666)),
    "ulivo": Rectangle(Segment(352, 4422), Segment(1595, 6380)),
}


def main() -> None:
    leaves = os.listdir("./dataset")
    for leaf in leaves:
        imgs = os.listdir(f"./dataset/{leaf}")
        # if leaf != "liquidambar":
        #     continue

        for img_name in imgs:
            img = cv2.imread(f"./dataset/{leaf}/{img_name}")
            print(f"./dataset/{leaf}/{img_name}")

            roi = rois[leaf]  # TODO compute it correctly
            h = find_leaf_height(img, roi)

            if leaf == "liquidambar":
                widths = get_leaf_widths(img, roi)
                leaf_roi = get_leaf_roi(img, roi, widths, h)
            else:
                widths = get_leaf_widths(img, roi, h)
                leaf_roi = get_leaf_roi(img, roi, widths, h)

            # Save into a test folder the vertical ROI of each dataset image
            for i in range(0, 11):
                row = int(h.corner + i * 0.1 * h.length)
                vert = Segment(row, 1)
                img = draw_rectangle(img, Rectangle(widths[i], vert), (0, 0, 255), 5)
                img = draw_rectangle(img, leaf_roi, (255, 0, 0), 5)

            cv2.imwrite(f"test/{leaf}.jpg", img)

            break
            return



if __name__ == "__main__":
    main()
