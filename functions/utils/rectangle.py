from __future__ import annotations
from functions.utils.segment import Segment


class Rectangle:
    def __init__(self, horiz: Segment, vert: Segment) -> None:
        """
        Creates a new rectangle, given the segments that represent its
        horizontal and vertical sizes

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - horiz: the horizontal size
        - vert: the vertical size
        """
        self.horiz = horiz
        self.vert = vert

    @classmethod
    def from_values(
        cls, topleft_row: int, topleft_col: int, width: int, height: int
    ) -> Rectangle:
        """
        Creates a new rectangle from the topleft coordinates, and the desired
        height and width

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - topleft_row: the vertical offset of the rectangle from the top
        - topleft_col: the horizontal offset of the rectangle from the left
        - width: the width of the rectangle
        - height: the height of the rectangle

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The rectangle itself
        """
        return cls(Segment(topleft_col, width), Segment(topleft_row, height))

    @classmethod
    def from_JSON(cls, details: dict[str, dict[str, int]]) -> Rectangle:
        """
        Creates a new segment from a dictionary representation of it

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - dict: a dictionary formatted as {corner: _float_, length: _float_}
            that represents a segment

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The same segment, but as a Segment class
        """

        if ("horiz" not in details) or ("vert" not in details):
            raise Exception("Invalid JSON format")

        return Rectangle(
            Segment.from_JSON(details["horiz"]), Segment.from_JSON(details["vert"])
        )

    def __repr__(self) -> str:
        """
        Describes the segment as
        "Rectangle[
                horiz: _horizontal_ _segment_ ,
            vert: _vertical_ _segment_
        ]"

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The string describing the rectangle
        """
        return f"Rectangle[\n\thoriz: {self.horiz}, \n\tvert: {self.vert}\n]"

    def get_horiz(self) -> Segment:
        """
        Returns the horizontal segment of the rectangle

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The horizontal segment of the rectangle
        """
        return self.horiz

    def get_vert(self) -> Segment:
        """
        Returns the vertical segment of the rectangle

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The vertical segment of the rectangle
        """
        return self.vert

    def to_JSON(self) -> dict[str, dict[str, int]]:
        """
        Converts the rectangle to a JSON object

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The rectangle as JSON object
        """
        return {"horiz": self.horiz.to_JSON(), "vert": self.vert.to_JSON()}
