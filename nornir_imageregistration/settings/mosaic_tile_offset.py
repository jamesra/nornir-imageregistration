"""
Created on Aug 31, 2022

@author: u0490822
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from typing import Any


class TileOffset(object):
    """A known offset from tile A to tile B: B's position minus A's, (Y, X) in full-resolution pixels."""

    @property
    def ID(self):
        return self._A, self._B

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Comment(self):
        return self._Comment

    @property
    def Offset(self):
        return self._Y, self._X

    def __init__(self, A, B, Y, X, Comment=None):
        # The offset runs from A to B, so swapping the IDs must negate it. The loader used to
        # swap A and B without negating, which reversed the offset of any row written B-first.
        if A > B:
            A, B, Y, X = B, A, -Y, -X
        self._A = A
        self._B = B
        self._X = X
        self._Y = Y
        self._Comment = Comment

    @staticmethod
    def FromValue(value: Any) -> TileOffset:
        """Accept a TileOffset, an (A, B, Y, X[, comment]) sequence, or a dict with those keys.

        Translate settings JSON can carry known_offsets as plain lists or objects.
        """
        if isinstance(value, TileOffset):
            return value
        if isinstance(value, dict):
            return TileOffset(A=int(value['A']), B=int(value['B']), Y=float(value['Y']), X=float(value['X']),
                              Comment=value.get('comment', value.get('Comment')))
        values = list(value)
        return TileOffset(A=int(values[0]), B=int(values[1]), Y=float(values[2]), X=float(values[3]),
                          Comment=values[4] if len(values) > 4 else None)

    def __str__(self):
        return f"{self.A:4} {self.B:4} {self.Y:8.1f} {self.X:8.1f}{'' if self.Comment is None else ' ' + self.Comment}\n"

    def __eq__(self, other):
        return self.A == other.A and self.B == other.B and self.X == other.X and self.Y == other.Y and self.Comment == other.Comment

    def __ge__(self, other):
        return self.ID.__ge__(other.ID)

    def __gt__(self, other):
        return self.ID.__gt__(other.ID)

    def __le__(self, other):
        return self.ID.__le__(other.ID)

    def __lt__(self, other):
        return self.ID.__lt__(other.ID)


def LoadMosaicOffsets(path: str):
    """Load tile offsets from a text file; returns list of TileOffset."""
    offsets = []

    with open(path, 'r') as offsets_file:
        csvReader = csv.reader(offsets_file, delimiter=' ', skipinitialspace=True)
        for (line_number, line) in enumerate(csvReader):
            if len(line) == 0:
                continue

            if line[0].startswith('#'):
                continue

            # Skip the header without printing a parsing error
            if line_number == 0 and line[0].startswith('A'):
                continue

            try:
                A = int(line[0])
                B = int(line[1])
                # TileOffset puts the lower ID first and negates the offset to match.
                Y = float(line[2])
                X = float(line[3])

                comment = None
                if len(line) >= 5:
                    if len(line[4]) > 0 and line[4] != "None":
                        comment = ' '.join(line[4:])

                o = TileOffset(A=A, B=B, X=X, Y=Y, Comment=comment)
                offsets.append(o)
            except Exception as e:
                print(f"Could not parse histogram line #{line_number}: {' '.join(line)}")
                print(f"{e}")

    return offsets


def SaveMosaicOffsets(offsets: list[TileOffset] | None, path: str):
    """Write tile offsets to a text file. Returns None."""
    if offsets is None:
        offsets = []

    sorted_input = sorted(offsets)

    with open(path, 'w') as offsets_file:
        # Rows are written Y before X; the header said "X Y" until 2026-11.
        header = f"{'A':4} {'B':4} {'Y':8} {'X':8} Comment\n"
        offsets_file.write(header)
        for o in sorted_input:
            output = f"{o.A:4} {o.B:4} {o.Y:8.1f} {o.X:8.1f} {'' if o.Comment is None else o.Comment}\n"
            offsets_file.write(output)


def MosaicOffsetsChecksum(offsets: Iterable[TileOffset] | None) -> str:
    """Checksum of the offsets' content (IDs and values, not comments or file formatting).

    Lets translate tell whether the manual offsets changed or went missing since a mosaic was
    built, which a file timestamp or a checksum of the file's path cannot (#219).
    """
    rows = sorted((o.A, o.B, round(float(o.Y), 1), round(float(o.X), 1)) for o in (offsets or []))
    text = '\n'.join(f"{a} {b} {y:.1f} {x:.1f}" for a, b, y, x in rows)
    return hashlib.md5(text.encode('ascii')).hexdigest()


if __name__ == '__main__':
    pass
