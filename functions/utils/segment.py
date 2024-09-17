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

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The segment itself
        """
        self.corner = corner
        self.length = length

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
        Describes the segment as "Segment[from: <corner>, long: <length>]"

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
