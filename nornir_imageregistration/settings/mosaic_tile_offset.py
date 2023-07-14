"""
Created on Aug 31, 2022

@author: u0490822
"""

import csv


class TileOffset(object):

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
        self._A = A if A < B else B
        self._B = B if A < B else A
        self._X = X
        self._Y = Y
        self._Comment = Comment

    def __str__(self):
        return f"{self.A:4} {self.B:4} {self.Y:8.1f} {self.X:8.1f}{'' if self.Comment is None else ' ' + self.Comment}\n"

    def __eq__(self, other):
        return self.A == other.A and self.B == other.B and self.X == other.X and self.Y == other.Y and self.Comment == other.Comment

    def __ne__(self, other):
        return not self.__eq__(other.ID)

    def __ge__(self, other):
        return self.ID.__ge__(other.ID)

    def __gt__(self, other):
        return self.ID.__gt__(other.ID)

    def __le__(self, other):
        return self.ID.__le__(other.ID)

    def __lt__(self, other):
        return self.ID.__lt__(other.ID)


def LoadMosaicOffsets(path: str):
    offsets = []

    with open(path, 'r') as offsets_file:
        csvReader = csv.reader(offsets_file, delimiter=' ', skipinitialspace=True, dialect=csv.Dialect.skipinitialspace)
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

                if A > B:  # Swap the IDs so the lowest ID is in A position
                    t = A
                    A = B
                    B = t

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
    if offsets is None:
        offsets = []

    sorted_input = sorted(offsets)

    with open(path, 'w') as offsets_file:
        header = f"{'A':4} {'B':4} {'X':8} {'Y':8} Comment\n"
        offsets_file.write(header)
        for o in sorted_input:
            A = o.A if o.A < o.B else o.B
            B = o.B if o.A < o.B else o.A
            output = f"{A:4} {B:4} {o.Y:8.1f} {o.X:8.1f} {'' if o.Comment is None else o.Comment}\n"
            offsets_file.write(output)


if __name__ == '__main__':
    pass
