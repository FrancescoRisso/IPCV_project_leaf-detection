import os
import cv2
import matplotlib.pyplot as plt

from functions.utils.rectangle import Rectangle
from functions.utils.segment import Segment

from functions.sharp_border import fourier_transform

from functions.utils.image import crop_image


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

        for img_name in imgs:
            img = cv2.imread(f"./dataset/{leaf}/{img_name}")
            print(f"./dataset/{leaf}/{img_name}")

            img = crop_image(img, rois[leaf])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = img[:, :, 1]
            img = cv2.threshold(sat, 35, 255, cv2.THRESH_BINARY)[1]

            plt.subplot(2, 3, (pos[leaf]))
            plt.title(leaf)
            plt.plot(fourier_transform(img))

            break
            return

    plt.show()
    input("Done")


if __name__ == "__main__":
    main()
