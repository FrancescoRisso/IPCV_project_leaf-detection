from __future__ import annotations

from cv2.typing import MatLike
from typing import Optional, Any

import json

from functions.lengths.px_size import get_px_height_in_mm, get_px_width_in_mm


class ImageFeatures:
    """
    An ImageFeature is an object that stores an image, and allows you to
    compute all its features.
    The features are cached, stored in attributes of the class, if
    already computed. The various get methods will check if the value is
    present, and if it is not, they will call the appropriate function,
    store the result and return it.
    It is also possible to store an ImageFeatures to a file, and to load
    it from a file.

    ---------------------------------------------------------------------
    What to do when adding a new feature/internal measure to an image:
    - add it as an attribute, either in the "internal values" or in the
        "model features" section. It must be an Optional[type]
    - add the getters for the value, that also update the attribute and
        set self.__modified to True if the value was changed
    - add the getter to the correct location in the dict in to_JSON
    - add a parser in load_details_from_file
    """

    def __init__(self, img: MatLike) -> None:
        # Image
        self.__img = img

        # Modified flag
        self.__modified: bool = False

        # Internal values
        self.__px_width_in_mm: Optional[float] = None
        self.__px_height_in_mm: Optional[float] = None

        # Model features

    def to_JSON(self) -> str:
        res: dict[str, dict[str, Any]] = {
            "features": {},
            "internal": {
                "px_width_in_mm": self.__get_px_width_in_mm(),
                "px_height_in_mm": self.__get_px_height_in_mm(),
            },
        }

        return json.dumps(res)

    def load_details_from_file(self, path: str) -> ImageFeatures:
        """
        Given an existing ImageFeatures and the path of the corresponding
        json file, updates the attributes with the values stored in the json
        file, leaving None to what is not present in the json file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: the path to the json file

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The ImageFeatures itself, to be able to do method chaining
        """
        with open(path, "r") as file:
            data = json.load(file)

        internals, features = data["internal"], data["features"]

        if internals.get("px_width_in_mm", None):
            self.__px_width_in_mm = internals["px_width_in_mm"]

        if internals.get("px_height_in_mm", None):
            self.__px_height_in_mm = internals["px_height_in_mm"]

        return self

    def store_to_file(self, path: str, force: bool = False) -> None:
        """
        Stores all the data to a file, in json format.
        If all the values were already loaded from a file (no recomputation),
        the file will not be written by default.
        The write can occur also if all the values were loaded, if the force
        flag is set.

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: the path of the file where to write
        - force: if set, the file will be written regardless of whether the
            values were computed or not
        """

        result = self.to_JSON()

        if force or self.__modified:
            with open(path, "w") as f:
                f.write(result)

    def __get_px_width_in_mm(self) -> float:
        if self.__px_width_in_mm:
            return self.__px_width_in_mm

        self.__px_width_in_mm = get_px_width_in_mm(self.__img)
        self.__modified = True
        return self.__px_width_in_mm

    def __get_px_height_in_mm(self) -> float:
        if self.__px_height_in_mm:
            return self.__px_height_in_mm

        self.__px_height_in_mm = get_px_height_in_mm(self.__img)
        self.__modified = True
        return self.__px_height_in_mm