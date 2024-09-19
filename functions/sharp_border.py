from cv2.typing import MatLike
import cv2

import numpy as np

def fourier_transform(img: MatLike) -> MatLike:
# def fourier_transform(img: MatLike) -> tuple[MatLike, MatLike]:
    """
    

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - MUST BE GRAYSCALE

    ---------------------------------------------------------------------
    OUTPUT
    ------
    
    """

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.array(np.float32(img), dtype=np.float32)
    # compl = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    # compl = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # compl = np.fft.fftshift(compl)

    freq = np.fft.fft2(img)

    # mag, ph = cv2.cartToPolar(compl[:, :, 0], compl[:, :, 1])

    return freq
