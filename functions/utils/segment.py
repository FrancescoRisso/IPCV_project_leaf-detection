from __future__ import annotations


class Segment:
    def __init__(self, corner: int, length: int) -> None:
        """
        Creates a new monodirectional segment, given its starting point and
        length

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - corner: the starting point of the segment
        - length: the length of the segment
        """
        self.corner = corner
        self.length = length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return (self.corner == other.corner) and (self.length == other.length)

    @classmethod
    def from_JSON(cls, details: dict[str, int]) -> Segment:
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

        if ("corner" not in details) or ("length" not in details):
            raise Exception("Invalid JSON format")

        return Segment(details["corner"], details["length"])

    def other_corner(self) -> int:
        """
        Returns the (monodirectional) coordinate of the end point of the
        segment

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The coordinate of the end point of the segment
        """
        return self.corner + self.length

    def intersect(self, other: Segment) -> Segment:
        """
        Returns a new segment that is the intersection between the current
        segment and another one

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - other: the other segment to intersect with

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The intersection
        """

        corner = max(self.corner, other.corner)
        other_corner = min(self.other_corner(), other.other_corner())
        return Segment(corner, other_corner - corner)

    def __repr__(self) -> str:
        """
        Describes the segment as "Segment[from: _corner_, long: _length_]"

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The string describing the segment
        """
        return f"Segment[from: {self.corner}, long: {self.length}]"

    def middle(self) -> int:
        """
        Computes the middle point of the segment, eventually rounding it down

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The coordinate of the midpoint
        """
        return self.corner + int(self.length / 2)

    def first_half(self) -> Segment:
        """
        Creates a new segment as the first half of the current one

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The first half of this segment
        """
        return Segment(self.corner, int(self.length / 2))

    def second_half(self) -> Segment:
        """
        Creates a new segment as the second half of the current one

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The second half of this segment
        """
        corner = self.corner + int(self.length / 2)
        length = self.length - int(self.length / 2)
        return Segment(corner, length)

    def other_half(self, half: Segment) -> Segment:
        """
        Given either the first or the second half of this segment, returns
        the other half

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The other half of this segment
        """
        if half == self.first_half():
            return self.second_half()
        elif half == self.second_half():
            return self.first_half()
        else:
            raise Exception(f"{half} is not the first or second half of {self}")

    def to_JSON(self) -> dict[str, int]:
        """
        Converts the segment to a JSON object

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The segment as JSON object (a dict)
        """
        return {"corner": int(self.corner), "length": int(self.length)}
