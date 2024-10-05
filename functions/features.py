from __future__ import annotations

from cv2.typing import MatLike
from typing import Optional, Any
from custom_types.tuple_of_11 import tuple_of_11
from custom_types.tuple_of_11 import to_tuple_of_11
from custom_types.tuple_of_11 import tuple_of_11_to_python_tuple


import json
import cv2

from functions.utils.rectangle import Rectangle
from functions.utils.segment import Segment

from functions.lengths.px_size import get_px_size
from functions.lengths.paper_roi import find_roi_boundaries, roi_boundaries_as_rect
from functions.lengths.leaf_height import find_leaf_height
from functions.lengths.leaf_width import get_leaf_widths


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
    - for any ImageFeature.__get... you call in the getter you added,
        go in that function and set as None the attribute you are working
        on, in order to ensure that your value is not cached if a
        dependency is changed
    - add the getter to the correct location in the dict in to_JSON
    - add a parser in load_details_from_file
    """

    def __init__(self, img: MatLike) -> None:
        # Image, in BGR
        self.__img = img

        # Modified flag
        self.__modified: bool = False

        # Internal values
        self.__px_width_in_mm: Optional[float] = None
        self.__px_height_in_mm: Optional[float] = None
        self.__paper_roi: Optional[Rectangle] = None
        self.__height_segment: Optional[Segment] = None
        self.__widths_segments: Optional[tuple_of_11[Segment]] = None

        # Model features
        self.__height: Optional[float] = None
        self.__width_0_perc_h: Optional[tuple_of_11[float]] = None

    def to_JSON(self) -> str:
        width_segments = tuple_of_11_to_python_tuple(self.__get_widths_segments())
        width_segments_json = [w.to_JSON() for w in width_segments]

        res: dict[str, dict[str, Any]] = {
            "features": {"height": self.__get_leaf_height()},
            "internal": {
                "px_width_in_mm": self.__get_px_width_in_mm(),
                "px_height_in_mm": self.__get_px_height_in_mm(),
                "paper_roi": self.__get_paper_roi().to_JSON(),
                "height_segment": self.__get_leaf_height_segment().to_JSON(),
                "widths": width_segments_json,
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

        if internals.get("paper_roi", None):
            self.__paper_roi = Rectangle.from_JSON(internals["paper_roi"])

        if internals.get("height_segment", None):
            self.__height_segment = Segment.from_JSON(internals["height_segment"])

        if features.get("height", None):
            self.__height = features["height"]

        if internals.get("widths", None):
            self.__widths_segments = to_tuple_of_11(
                [Segment.from_JSON(segm) for segm in internals["widths"]]
            )

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

        self.__px_width_in_mm = get_px_size(
            cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV), self.__get_paper_roi(), False
        )
        self.__modified = True
        return self.__px_width_in_mm

    def __get_px_height_in_mm(self) -> float:
        if self.__px_height_in_mm:
            return self.__px_height_in_mm

        self.__px_height_in_mm = get_px_size(
            cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV), self.__get_paper_roi(), True
        )
        self.__modified = True
        self.__height = None
        return self.__px_height_in_mm

    def __get_paper_roi(self) -> Rectangle:
        if self.__paper_roi:
            return self.__paper_roi

        self.__paper_roi = roi_boundaries_as_rect(find_roi_boundaries(self.__img))
        self.__modified = True
        self.__px_width_in_mm = None
        self.__px_height_in_mm = None
        self.__height_segment = None
        self.__widths_segments = None
        return self.__paper_roi

    def __get_leaf_height_segment(self) -> Segment:
        if self.__height_segment:
            return self.__height_segment

        self.__height_segment = find_leaf_height(self.__img, self.__get_paper_roi())
        self.__modified = True
        self.__height = None
        self.__widths_segments = None
        return self.__height_segment

    def __get_leaf_height(self) -> float:
        if self.__height:
            return self.__height

        height_px = self.__get_leaf_height_segment().length
        self.__height = height_px * self.__get_px_height_in_mm()
        self.__modified = True
        return self.__height

    def __get_widths_segments(self) -> tuple_of_11[Segment]:
        if self.__widths_segments:
            return self.__widths_segments

        self.__widths_segments = get_leaf_widths(
            self.__img, self.__get_paper_roi(), self.__get_leaf_height_segment()
        )
        self.__modified = True
        return self.__widths_segments
