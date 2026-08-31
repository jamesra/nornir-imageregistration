from __future__ import annotations
import collections
import copy
import logging
import os
import warnings
from operator import itemgetter

import numpy as np
from numpy.typing import NDArray
import scipy

from collections.abc import Iterable, Sequence

import nornir_imageregistration
from nornir_imageregistration.spatial_distance import cdist as pairwise_cdist
from nornir_imageregistration.tile_overlap import TileOverlap
import nornir_imageregistration.transforms
import nornir_imageregistration.type_info
import nornir_pools
import nornir_shared.prettyoutput as prettyoutput
from typing import cast

ID_Value = collections.namedtuple('ID_Magnitude', ['ID', 'Value'])

TileOffset = collections.namedtuple('TileOffset', ('A', 'B', 'Y', 'X'))


def _sort_array_on_column(a, iCol, ascending=False):
    """Sort a 2D array by the values in the specified column.

    :param a: 2D array to sort.
    :param iCol: Column index to sort by.
    :param ascending: If True, ascending order; otherwise descending.
    :return: Sorted array (same shape as a).
    """
    iSorted = np.argsort(a[:, iCol], 0)
    if not ascending:
        iSorted = np.flipud(iSorted)
    return a[iSorted, :]


def create_pair_id(A: int | Sequence[int] | tuple[int, int] | LayoutPosition,
                   B: int | LayoutPosition | None = None) -> tuple[int, int]:
    """Form a canonical pair ID (smaller id first, both as integers).

    :param A: First ID or (A, B) sequence/tuple when B is None.
    :param B: Second ID; must be None if A is a sequence or tuple of two IDs.
    :return: Tuple (min_id, max_id) as integers.
    """

    if isinstance(A, tuple):
        if B is not None:
            raise ValueError("B must not be specified if A is a tuple")
        return int(A[0]), int(A[1])
    if isinstance(A, Sequence) and B is None:
        if len(A) < 2:
            raise ValueError("A sequence input must contain at least two items")
        B = A[1]
        A = A[0]
    elif isinstance(A, Iterable):
        seqA = list(A)
        if len(seqA) < 2:
            raise ValueError("An iterable input must contain at least two items")
        B = seqA[1]
        A = seqA[0]
    elif isinstance(A, int):
        pass
    else:
        raise ValueError("Invalid type for A: {0}".format(type(A)))

    a_id = A.ID if isinstance(A, LayoutPosition) else A
    b_id = B.ID if isinstance(B, LayoutPosition) else B

    if a_id is None or b_id is None:
        raise ValueError("Both A and B IDs must be defined")
    a_id = int(a_id)
    b_id = int(b_id)

    if a_id < b_id:
        return a_id, b_id
    else:
        return b_id, a_id


class LayoutPosition:
    """This is an anchor with a number of springs of a certain length attached.  In our use the anchor is a tile and the spring size
       and strength is determined by the offset to overlap an adjacent tile

       Offsets is a numpy array of the form [[ID Y X Weight]]
    """

    iOffsetID: int = 0
    iOffsetY: int = 1
    iOffsetX: int = 2
    iOffsetWeight: int = 3

    _ID: int
    _position: NDArray[np.floating]
    _OffsetArray: NDArray[np.float64]
    _dims: nornir_imageregistration.type_info.RectLike | None
    _IDToIndex: dict[int, int] | None = None

    _connected_id_cache: NDArray[np.integer] | None = None
    _irow_cache: NDArray[np.integer] | None = None

    # offset_dtype = np.dtype([('ID', int), ('Y', float), ('X', float), ('Weight', float)])

    @property
    def ID(self) -> int:
        return self._ID

    @property
    def Position(self) -> NDArray[np.floating]:
        """Our position in the layout"""
        return self._position

    @Position.setter
    def Position(self, value: NDArray[np.floating] | Iterable[np.floating]):
        """Our position in the layout"""
        if not isinstance(value, np.ndarray):
            self._position = np.array(value, dtype=np.float64)
        else:
            self._position = value.astype(np.float64, copy=False)

        assert (self._position.ndim == 1)
        return

    @property
    def OffsetArray(self) -> NDArray[np.float64]:
        """
        Read-only use please.
        Each row is [ID Y X Weight]
        """
        readonly_array = np.array(self._OffsetArray)
        readonly_array.setflags(write=False)
        return readonly_array

    @property
    def IsIsolated(self) -> bool:
        """Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no offsets"""
        return len(self._OffsetArray) == 0

    @property
    def Weights(self) -> NDArray[np.floating]:
        return self._OffsetArray[:, LayoutPosition.iOffsetWeight]

    @Weights.setter
    def Weights(self, value: NDArray[np.floating] | float):
        self._OffsetArray[:, LayoutPosition.iOffsetWeight] = value

    @property
    def ConnectedIDs(self) -> NDArray[np.integer]:
        if self._connected_id_cache is None:
            self._connected_id_cache = self._OffsetArray[:, LayoutPosition.iOffsetID].astype(int, copy=False)

        return self._connected_id_cache

    @property
    def NumConnections(self) -> int:
        return self._OffsetArray.shape[0]

    @property
    def dims(self):
        return self._dims

    def GetOffset(self, ID):
        iKnown = self.ConnectedIDs == ID
        return self._OffsetArray[iKnown, LayoutPosition.iOffsetY:LayoutPosition.iOffsetX + 1].flatten()

    def ContainsOffset(self, ID) -> bool:
        iKnown = self.ConnectedIDs == ID
        return bool(np.any(iKnown))

    def GetWeight(self, ID) -> float:
        iKnown = self.ConnectedIDs == ID
        # Boolean indexing yields a shape-(1,) array, and NumPy 2 refuses to convert
        # anything but a 0-d array to a scalar, so the unflattened float() raised
        # TypeError on every call. GetOffset above already flattens for this reason.
        return float(self._OffsetArray[iKnown, LayoutPosition.iOffsetWeight].flatten()[0])

    @property
    def IDToIndex(self) -> dict[int, int]:
        """
        Maps the ID of a connected node to the row index in the offset array
        :return:
        """
        if self._IDToIndex is None:
            self._IDToIndex = dict()
            for (i, ID) in enumerate(self._OffsetArray[:, 0]):
                self._IDToIndex[ID] = i

        return self._IDToIndex

    def SetOffset(self, ID: int, offset: nornir_imageregistration.type_info.PointLike, weight: float):
        """Set the offset for the specified Layout position ID.
           This means that when we subtract our position from the other ID's position we hope to obtain this offset value.
        """

        if np.isnan(weight):
            raise ValueError("weight is not a number")

        new_row = np.array((ID, offset[0], offset[1], weight))  # , dtype=LayoutPosition.offset_dtype, ndmin=2)
        iKnown = self.ConnectedIDs == ID
        if np.any(iKnown):
            # Update a row
            self._OffsetArray[iKnown] = new_row
        else:
            # Insert a new row 
            self._OffsetArray = np.vstack((self._OffsetArray, new_row))
            if self._OffsetArray.ndim == 1:
                self._OffsetArray = np.reshape(self._OffsetArray, (1, self._OffsetArray.shape[0]))
            else:
                self._OffsetArray = _sort_array_on_column(self._OffsetArray, 0, ascending=True)

            self._IDToIndex = None
            self._connected_id_cache = None
            self._irow_cache = None
        return

    def RemoveOffset(self, ID):
        """
        Remove the offset to the other tile entirely
        """

        iKnown = self.ConnectedIDs == ID
        if np.any(iKnown):
            self._OffsetArray = self._OffsetArray[iKnown == False, :]
            self._IDToIndex = None
            self._connected_id_cache = None
            self._irow_cache = None
        else:
            # `Warning(...)` built an exception instance and dropped it, so nothing was ever
            # emitted, and it sat outside this branch so it also ran on success. Logging
            # rather than warnings.warn: this is a runtime data condition, not API misuse,
            # and warnings.warn shows once per call site by default, which would hide
            # repeats. warnings.warn in this module is reserved for the deprecation at
            # ToMosaic. Measured on healthy 4- and 9-tile mosaics this never fires, so it is
            # a real signal rather than noise -- RemoveOverlap removes both directions, so a
            # pair whose nodes exist without an offset logs once per direction (#129).
            logging.getLogger(__name__).warning(
                'Removing non-existent offset: %s->%s', self.ID, ID)
        return

    def get_row_indicies(self, connected_nodes: Sequence[LayoutPosition] | None = None) -> NDArray[np.integer]:
        """
        Given a set of connected nodes, return the index into our _OffsetArray
        :return: A numpy array of row indices
        """
        if connected_nodes is None:
            if self._irow_cache is None:
                self._irow_cache = np.array(range(0, len(self.ConnectedIDs)), dtype=int)
            return self._irow_cache
        else:
            # connected_IDs = [n.ID for n in connected_nodes]
            return np.array([self.IDToIndex[n.ID] for n in
                             connected_nodes])  # nornir_imageregistration.IndexOfValues(self.ConnectedIDs, connected_IDs)

    get_row_indices = get_row_indicies  # alias with correct spelling

    def TensionVectors(self, connected_nodes: Sequence[LayoutPosition] | None = None) -> NDArray[np.floating]:
        """The difference between the current connected_positions and the expected positions based on our offsets
        :param connected_nodes:
        :param ndarray connected_nodes: [ID Y X] Position of the connected nodes"""
        if connected_nodes is None or len(connected_nodes) == 0:
            return np.zeros((1, 2), dtype=np.float64)

        connected_positions = np.vstack([n.Position for n in connected_nodes])
        relative_connected_positions = connected_positions - self.Position
        iRows = self.get_row_indices(connected_nodes)

        return relative_connected_positions - self._OffsetArray[iRows,
                                              LayoutPosition.iOffsetY:LayoutPosition.iOffsetX + 1]

    def NetTensionVector(self, connected_nodes: Sequence[LayoutPosition]) -> NDArray[np.floating]:
        """
        A set of N rows indicating where this node needs to move to have no tension with the linked node on that row
        """
        position_difference = self.TensionVectors(connected_nodes)
        return np.sum(position_difference, 0)

    def WeightedNetTensionVector(self, connected_nodes: Sequence[LayoutPosition]) -> NDArray[np.floating]:
        """The direction of the vector this tile wants to move after summing all     of the offsets
        :param connected_nodes:
        :param ndarray connected_nodes: Position of the connected nodes"""
        if len(connected_nodes) == 0:
            return np.zeros((1, 2), dtype=np.float64)

        position_difference = self.TensionVectors(connected_nodes)

        # Cannot weight more than 1.0
        # normalized_weight = self._OffsetArray[:,LayoutPosition.iOffsetWeight] / np.max(self._OffsetArray[:,LayoutPosition.iOffsetWeight])
        iRows = self.get_row_indices(connected_nodes)
        weights = self._OffsetArray[iRows, LayoutPosition.iOffsetWeight]
        total_weight = np.sum(weights)
        if total_weight != 0:
            normalized_weight = weights / total_weight
        else:
            normalized_weight = weights

        assert (np.all(weights >= 0))
        assert (np.all(weights <= 1.0))
        # assert(np.sum(normalized_weight) == 1.0)
        weighted_position_difference = position_difference * normalized_weight.reshape((normalized_weight.shape[0], 1))

        return np.sum(weighted_position_difference, 0)

    def MaxTensionVector(self, connected_nodes: Sequence[LayoutPosition]) -> ID_Value:
        """
        The largest tension vector
        :return: tuple of (ID, magnitude) of the largest tension vector
        """
        if len(connected_nodes) == 0:
            return ID_Value(None, np.array((0, 0)))

        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_max_tension = magnitudes.argmax()
        return ID_Value(self.OffsetArray[i_max_tension, self.iOffsetID], position_difference[i_max_tension, :])

    def MinTensionVector(self, connected_nodes: Sequence[LayoutPosition]) -> ID_Value:
        """
        The smallest tension vector
        :return: tuple of (ID, magnitude) of the smallest tension vector
        """
        if len(connected_nodes) == 0:
            return ID_Value(None, np.array((0, 0)))

        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_min_tension = magnitudes.argmin()
        return ID_Value(self.OffsetArray[i_min_tension, self.iOffsetID], position_difference[i_min_tension, :])

    def MaxTensionMagnitude(self, connected_nodes: Sequence[LayoutPosition]) -> ID_Value:
        """
        The largest tension vector
        :return: tuple of (ID, magnitude) of the largest tension vector
        """
        if len(connected_nodes) == 0:
            return ID_Value(None, 0)

        position_difference = self.MaxTensionVector(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference.Value ** 2, 1))
        i_max_tension = magnitudes.argmax()
        return ID_Value(self.OffsetArray[i_max_tension, self.iOffsetID], magnitudes[i_max_tension])

    def MinTensionMagnitude(self, connected_nodes: Sequence[LayoutPosition]) -> ID_Value:
        """
        The smallest tension vector
        :return: tuple of (ID, magnitude) of the smallest tension vector
        """
        if len(connected_nodes) == 0:
            return ID_Value(None, 0)

        position_difference = self.TensionVectors(connected_nodes)
        magnitudes = np.sqrt(np.sum(position_difference ** 2, 1))
        i_min_tension = magnitudes.argmin()
        return ID_Value(self.OffsetArray[i_min_tension, self.iOffsetID], magnitudes[i_min_tension])

    def ScaleOffsetWeightsByPosition(self, connected_nodes: Sequence[LayoutPosition]):
        """
        Reweight our set of weights based on how far from this expectation our offsets are.  THis is useful if we believe our initial positions are largely accurate but
        our calculated desired offsets may have errors.
        :param ndarray connected_nodes: The locations we believe our connected positions should be.
        :raises NotImplementedError: Always.  This function has never been implemented; the
            docstring above records the intended semantics for whoever finishes it.
        """
        # The raise used to sit *after* the weight computation, so a caller saw whatever that
        # computation did first rather than the author's message. Under this module's numpy
        # error state (divide='raise', invalid='raise') a layout whose offsets agree with its
        # positions has zero tension everywhere, so medianDistance is zero and distance /
        # medianDistance is 0/0 -- surfacing as "FloatingPointError: invalid value encountered
        # in divide", which reads like an arithmetic bug in a working function instead of an
        # unfinished one. Raising first reports the actual state for every input. The
        # unreachable computation and assignment that followed are gone; git history has them
        # if the intended formula is wanted. (#250)
        raise NotImplementedError(
            "ScaleOffsetWeightsByPosition has never been implemented; it needs to accept a "
            "LayoutPosition list as its argument. See #250 before relying on it.")

    def __init__(self,
                 ID: int,
                 position: nornir_imageregistration.type_info.PointLike,
                 dims: nornir_imageregistration.type_info.RectLike | None = None,
                 *args, **kwargs):
        """
        :param int ID: ID number
        :param tuple position: Center position (Y,X). Accepts NumPy or CuPy; stored as NumPy.
        :param tuple dims: Dimensions of node (Y,X)
        """
        if not isinstance(ID, int):
            raise TypeError("Node ID must be an integer: {0}".format(ID))

        self._ID = ID
        self.Position = nornir_imageregistration.EnsureNumpyArray(position, dtype=np.float64)
        self._OffsetArray = np.empty((0, 4), dtype=np.float64)  # dtype=LayoutPosition.offset_dtype)
        self._dims = dims
        self._IDToIndex = None

        self._connected_id_cache = None
        self._irow_cache = None

    def __eq__(self, other: LayoutPosition) -> bool:
        if isinstance(other, LayoutPosition):
            return self._ID == other.ID  # change that to your needs 

        return False

    def __ne__(self, other: LayoutPosition) -> bool:
        if isinstance(other, LayoutPosition):
            return self._ID != other.ID  # change that to your needs 

        return True

    def __hash__(self) -> int:
        return self._ID

    def copy(self) -> LayoutPosition:
        """:return: A copy of the object"""
        c = LayoutPosition(self._ID,
                           position=self.Position.copy())
        c._OffsetArray = self._OffsetArray.copy()
        return c

    def __str__(self) -> str:
        return f"{self._ID} y:{self.Position:f2} x:{self.Position:f2}"


OverlapKeyType = TileOverlap | tuple[int, int]


class Layout:
    """ Records the optimal offset from each tile in a mosaic tile to overlapping tiles.
        IDs of nodes should be incremental and match the row index of the array."""

    # Offsets into node position array
    iNodeID: int = 0
    iNodeY: int = 1
    iNodeX: int = 2

    ID: int
    _nodes: dict[int, LayoutPosition]

    @classmethod
    def _parameter_to_offset_IDs(cls, param: OverlapKeyType) -> tuple[int, int]:
        """
        :param param: Either a TileOverlap object or a tuple of node ID's.
        :return: A tuple of node ID's
        """
        if isinstance(param, nornir_imageregistration.tile_overlap.TileOverlap):
            return param.A.ID, param.B.ID
        else:
            return param

    @property
    def nodes(self) -> dict[int, LayoutPosition]:
        """
        :return: A dictionary mapping ID to LayoutPosition objects
        """
        return self._nodes

    @property
    def linked_nodes(self) -> set[tuple[int, int]]:
        """
        :return: A set of tuples of linked IDs, lowest ID value in the first position
        """
        # Return the set of linked nodes
        pairs: set[tuple[int, int]] = set()
        for node in self.nodes.values():
            # pairs
            # for connected_ID in node.ConnectedIDs:
            node_pairs = [(min(node.ID, connected_ID), max(node.ID, connected_ID)) for connected_ID in node.ConnectedIDs]
            # node_pairs.append(tuple(sorted([node.ID, connected_ID])))

            pairs = pairs.union(node_pairs)

        return pairs

    @property
    def average_center(self) -> NDArray[np.floating]:
        """
        :return: The average of the center positions of all tiles in the layout
        """
        centers = [n.Position for n in self.nodes.values()]
        centers_stacked = np.vstack(centers)
        avg_center = np.average(centers_stacked, axis=0)
        return avg_center

    @property
    def MaxWeightedNetTensionMagnitude(self) -> ID_Value:
        """Returns the (ID, Magnitude) of the node with the largest weighted net tension vector."""
        net_tension_vectors = self.WeightedNetTensionVectors()
        tension_magnitude = nornir_imageregistration.array_distance(net_tension_vectors[:, 1:])
        i_max = np.argmax(tension_magnitude)
        return ID_Value(net_tension_vectors[i_max, 0], tension_magnitude[i_max])
        # return np.max(nornir_imageregistration.array_distance(net_tension_vectors))

    @property
    def MaxNetTensionMagnitude(self) -> ID_Value:
        """Returns the (ID, Magnitude) of the node with the largest net tension vector."""
        net_tension_vectors = self.NetTensionVectors()
        tension_magnitude = nornir_imageregistration.array_distance(net_tension_vectors[:, 1:])
        i_max = np.argmax(tension_magnitude)
        return ID_Value(net_tension_vectors[i_max, 0], tension_magnitude[i_max])

    @property
    def MaxTensionMagnitude(self) -> ID_Value | None:
        """
        The largest single tension between any two nodes in the layout
        :return: An array of (A,B,Magnitude) where A,B are IDs
        """

        tension_vectors = self.MaxTensionVectors
        tension_magnitude = nornir_imageregistration.array_distance(tension_vectors[:, 2:4])
        if len(tension_magnitude) == 0:
            return None

        i_max = tension_magnitude.argmax()
        pair = cast(tuple[int, int], tuple(np.asarray(tension_vectors[i_max, 0:2], dtype=int).tolist()))
        return ID_Value(create_pair_id(pair), tension_magnitude[i_max])

    @property
    def MinTensionMagnitude(self) -> ID_Value | None:
        """
        The smallest tension between any two nodes in the layout
        :return: A tuple of (A,B,Magnitude) where A,B are IDs
        """

        tension_vectors = self.MinTensionVectors
        tension_magnitude = nornir_imageregistration.array_distance(tension_vectors[:, 2:4])
        if len(tension_magnitude) == 0:
            return None

        i_min = tension_magnitude.argmin()
        pair = cast(tuple[int, int], tuple(np.asarray(tension_vectors[i_min, 0:2], dtype=int).tolist()))
        return ID_Value(create_pair_id(pair), tension_magnitude[i_min])

    @property
    def MinWeightedNetTensionMagnitude(self) -> ID_Value:
        """Returns the (ID, Magnitude) of the node with the largest weighted net tension vector."""
        net_tension_vectors = self.WeightedNetTensionVectors()
        tension_magnitude = nornir_imageregistration.array_distance(net_tension_vectors[:, 1:])
        i_min = np.argmin(tension_magnitude)
        return ID_Value(net_tension_vectors[i_min, 0], tension_magnitude[i_min])
        # return np.max(nornir_imageregistration.array_distance(net_tension_vectors))

    def __str__(self) -> str:
        return "Layout {0} nodes {1} Connections {2}".format(self.ID, len(self.nodes.keys()), len(self.linked_nodes))

    def Contains(self, ID: int) -> bool:
        """
        :rtype: bool
        :return: True if layout contains the ID
        """
        return ID in self._nodes.keys()

    def SetOffset(self, A_ID: int, B_ID: int, offset: NDArray[np.floating], weight: float = 1.0):
        """
        Specify the expected offset between two nodes in the spring model.
        """
        A = self.nodes[A_ID]
        B = self.nodes[B_ID]
        A.SetOffset(B.ID, offset, weight)
        B.SetOffset(A.ID, -offset, weight)

    def ContainsOffset(self, overlap: OverlapKeyType) -> bool:
        """:return: True if the layout has an offset between the two nodes"""
        (A_ID, B_ID) = Layout._parameter_to_offset_IDs(overlap)

        if not (self.Contains(A_ID) and self.Contains(B_ID)):
            return False

        A = self.nodes[A_ID]
        B = self.nodes[B_ID]

        return A.ContainsOffset(B_ID) and B.ContainsOffset(A_ID)

    def RemoveOverlap(self, overlap: OverlapKeyType):
        """

        :param overlap:
        :return:
        """
        (A_ID, B_ID) = Layout._parameter_to_offset_IDs(overlap)

        if self.Contains(A_ID) and self.Contains(B_ID):
            A = self.nodes[A_ID]
            B = self.nodes[B_ID]
            A.RemoveOffset(B.ID)
            B.RemoveOffset(A.ID)
        return

    def RemoveNode(self, node_ID: int) -> bool:
        if node_ID in self.nodes:
            node = self.nodes[node_ID]

            for connected_ID in node.ConnectedIDs:
                self.RemoveOverlap((node_ID, connected_ID))

            del self.nodes[node_ID]

            return True

        return False

    def GetPosition(self, ID: int):
        """Return the position array for a set of nodes, sorted by node ID"""
        return self.nodes[ID].Position

    def GetPositions(self, IDs: list[int] | int | NDArray[np.integer] | None = None) -> NDArray[np.floating]:
        """Return the position array for a set of nodes, sorted by node ID"""

        if IDs is None:
            IDs = list(self.nodes.keys())
            IDs.sort()

        elif isinstance(IDs, int):
            IDs = [IDs]

        normalized_ids = [int(tile_id) for tile_id in IDs]
        if len(normalized_ids) == 0:
            return np.empty((0, 2))

        positions = np.vstack([self.nodes[tileID].Position for tileID in normalized_ids])
        #
        # positions = np.empty((len(IDs), 2))
        # for i, tileID in enumerate(IDs):
        # positions[i,:] = self.nodes[tileID].Position

        return positions

    def GetNodes(self, IDs: list[int] | int | NDArray[np.integer] | None = None) -> list[LayoutPosition]:
        """Return the sorted subset of nodes by IDs as a list"""

        if IDs is None:
            IDs = sorted(self.nodes.keys())
        elif isinstance(IDs, int):
            IDs = [IDs]

        normalized_ids = [int(tile_id) for tile_id in IDs]
        nodes = [self.nodes[tileID] for tileID in normalized_ids]

        return nodes

    def GetOffsetWeightExtrema(self) -> tuple[float, float]:
        """
        :return: A tuple with the (min,max) weight values of offsets in the layout
        """

        maxWeight = np.nan
        minWeight = np.nan

        first = True
        for node in self._nodes.values():
            # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
            if node.IsIsolated:
                continue

            weights = node.OffsetArray[:, LayoutPosition.iOffsetWeight]

            if first:
                first = False
                minWeight = np.min(weights)
                maxWeight = np.max(weights)
            else:
                minWeight = min((minWeight, np.min(weights)))
                maxWeight = max((maxWeight, np.max(weights)))

        return minWeight, maxWeight

    def NetTensionVector(self, ID: int) -> NDArray[np.floating]:
        """Return the net tension vector of the specified ID"""

        node = self.nodes[ID]
        linked_nodes = self.GetNodes(node.ConnectedIDs)

        return node.NetTensionVector(linked_nodes)

    def NetTensionVectors(self) -> NDArray[np.floating]:
        """Return all net tension vectors for our nodes"""
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 3))
        for (i, ID) in enumerate(IDs):
            output[i, 0] = ID
            output[i, 1:] = self.NetTensionVector(ID)

        return output

    def PairTensionVector(self, A: LayoutPosition, B: LayoutPosition) -> NDArray[np.floating]:
        """Return the tension vector between A and B
        :return: The ideal offset between A and B
        """

        pair = create_pair_id(A, B)
        node = self.nodes[pair[0]]
        linked_nodes = self.GetNodes(pair[1])

        return node.NetTensionVector(linked_nodes)

    #     def PairTensionMagnitude(self, A, B):
    #         '''Return the tension vector between A and B
    #         :return: The ideal offset between A and B
    #         '''
    #
    #         pair = create_pair_id(A,B)
    #         node = self.nodes[pair[0]]
    #         linked_nodes = self.GetNodes(pair[1])
    #
    #         net = node.NetTensionVector(linked_nodes)

    def WeightedNetTensionVector(self, ID: int) -> NDArray[np.floating]:
        """Return the net tension vector of the specified ID"""

        node = self.nodes[ID]
        linked_node_positions = self.GetNodes(node.ConnectedIDs)

        return node.WeightedNetTensionVector(linked_node_positions)

    def WeightedNetTensionVectors(self) -> NDArray[np.floating]:
        """Return all net tension vectors for our nodes"""
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 3))
        for i, TileID in enumerate(IDs):
            output[i, 0] = TileID
            output[i, 1:] = self.WeightedNetTensionVector(TileID)

        return output

    @property
    def MaxTensionVectors(self) -> NDArray[np.floating]:
        """
        Return the maximum tension vector for each node
        """
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 4))
        i = 0
        for ID in IDs:
            node = self.nodes[ID]
            node_max = node.MaxTensionVector(self.GetNodes(node.ConnectedIDs))
            if node_max[0] is None:
                continue

            pair = create_pair_id(ID, node_max.ID)
            output[i, :] = (pair[0], pair[1], node_max[1][0], node_max[1][1])
            i += 1

        return np.array(output[0:i, :])

    @property
    def MinTensionVectors(self) -> NDArray[np.floating]:
        """
        Return the minimum tension vector for each node
        """
        IDs = list(self.nodes.keys())
        IDs.sort()
        output = np.zeros((len(IDs), 4))
        i = 0
        for ID in IDs:
            node = self.nodes[ID]
            node_min = node.MinTensionVector(self.GetNodes(node.ConnectedIDs))
            if node_min[0] is None:
                continue

            pair = create_pair_id(ID, node_min[0])
            output[i, :] = (pair[0], pair[1], node_min[1][0], node_min[1][1])
            i += 1

        return np.array(output[0:i, :])

    def CreateNode(self, ID, position, dims=None):
        """

        :param ID:
        :param position:
        :param dims:
        :return:
        """
        assert (not ID in self.nodes)
        node = LayoutPosition(ID, position, dims)
        self.nodes[ID] = node
        return

    def CreateOffsetNode(self, Existing_ID, New_ID, scaled_offset, Weight):
        """Add a new position to the layout.  Place the new relative to the specified existing position plus an offset"""

        new_position = self.GetPosition(Existing_ID) + scaled_offset
        self.CreateNode(New_ID, new_position)
        self.SetOffset(Existing_ID, New_ID, scaled_offset, Weight)
        return

    NextLayoutID = 0

    def __init__(self):

        self.ID = Layout.NextLayoutID
        Layout.NextLayoutID += 1
        self._nodes = {}
        return

    def copy(self):
        c = Layout()
        c._nodes = {n.ID: n.copy() for n in self._nodes.values()}
        return c

    def Translate(self, vector):
        """Move all nodes by offset"""
        for node in self.nodes.values():
            node.Position = node.Position + vector

        return

    def TranslateToZeroOrigin(self):
        """Translates the layout so the min X/Y of all tile positions is 0,0"""
        positions = self.GetPositions()
        origin_offset = np.min(positions, 0)
        self.Translate(-origin_offset)

    def Merge(self, layoutB):
        """Merge layout directly into our layout"""

        self.nodes.update(layoutB.copy().nodes)

    @classmethod
    def RelaxNodes(cls, layout_obj: Layout, vector_scalar: float | None = None):
        """Adjust the position of each node along its tension vector
        :param Layout layout_obj: The layout to relax
        :param float vector_scalar: Multiply the weighted tension vectors by this amount before adjusting the position.  A high value is faster but may not be constrained.  A low value is slower but safe.
        :return: nx2 array of (node ID, sort weight), one row per *connected* node.

        Isolated nodes are omitted from the returned array rather than left as zero
        rows. A zero row is indistinguishable from a real entry for node ID 0, and
        the movement loop below walks every row, so each isolated node used to make
        it relax node 0 an extra time -- measured as node 0 being visited 3 times
        for 2 isolated nodes, landing 2.2 px away from where it belonged -- or raise
        KeyError when the layout had no node 0 at all. Isolated nodes are expected
        here, not hypothetical: they are the documented result of prune.
        """

        # TODO: Get rid of vector scalar.  Instead calculate the net tension vector at the new position.  Then add them and apply the merged vector. 

        node_movement = np.zeros((len(layout_obj.nodes), 2))

        if vector_scalar is None:
            vector_scalar = 1.0

        # vectors = {}

        # min_tension_node_id = layout_obj.MinWeightedNetTensionMagnitude[0]  # The node with the least tension 
        # nodes = layout_obj.nodes.values()

        # Todo: Sort highest to lowest tension vectors, then adjust movement in that order
        connected_count = 0
        for node_id in layout_obj.nodes:
            node = layout_obj.nodes[node_id]
            if node.NumConnections == 0:
                continue

            vector = layout_obj.WeightedNetTensionVector(node.ID)

            weights = node.Weights
            weight_sum = np.sum(weights) / node.NumConnections
            magnitude = np.sqrt(vector.dot(vector))
            weight_sum *= magnitude  # If it wants to go a long ways, and has a high weight, I want to move it first
            # vectors[ID] = vector 
            node_movement[connected_count, 0] = node.ID
            node_movement[connected_count, 1] = weight_sum
            connected_count += 1

        # Drop the unused tail rather than the rows of whichever nodes were isolated;
        # the loop above packs connected nodes into the front of the array.
        node_movement = node_movement[:connected_count]

        sort_by_weight_asc = np.argsort(node_movement[:, 1])

        sorted_node_movement = node_movement[sort_by_weight_asc, 0]

        for i in range(int(sorted_node_movement.shape[0]) - 1, -1,
                       -1):  # Reversing the range calls saves me a np.flip and this function is a bottleneck
            node_id = int(sorted_node_movement[i])
            vector = layout_obj.WeightedNetTensionVector(node_id) * vector_scalar

            node = layout_obj.nodes[node_id]
            node.Position += vector

        # OK, move all of the nodes according to the net movement
        # for (i, node) in enumerate(nodes): 
        #     # Skip the node with the smallest amount of tension, the others can move around it as an anchor
        #     if i == min_tension_node_id:
        #         continue 
        #
        #     node.Position = node.Position + (node_movement[i, 1:3])

        return node_movement

    @classmethod
    def MergeLayouts(cls, layoutA, layoutB, offset):
        """
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        """

        layoutB.Translate(offset)
        layoutA.nodes.update(layoutB.nodes)
        return layoutA

    def _CreateTransform(self, ID, full_res_image_shape):
        """
        Create a transform for the position in the layout
        """
        # OriginalImageSize = (bounding_box[spatial.iRect.MaxY], bounding_box[spatial.iRect.MaxX])

        # return tfactory.CreateRigidMeshTransform(target_image_shape=OriginalImageSize,
        #                                      source_image_shape=OriginalImageSize,
        #                                      rangle=0,
        #                                      warped_offset=self.GetPosition(ID)) 
        return nornir_imageregistration.transforms.factory.CreateRigidTransform(target_image_shape=full_res_image_shape,
                                                                                source_image_shape=full_res_image_shape,
                                                                                rangle=0,
                                                                                warped_offset=self.GetPosition(ID))

    def ToTransforms(self, tiles):
        """
        Create a new set of transform for each tile in the tiles dictionary
        :param tiles: Dictionary of tile ID to tiles
        :return: sorted list of transforms for ID's found tiles
        """

        transforms = []

        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue

            tile = tiles[ID]

            transform = self._CreateTransform(ID,
                                              full_res_image_shape=tile.ImageSize * tile.image_to_source_space_scale)

            transforms.append(transform)

        return transforms

    def UpdateTileTransforms(self, tiles):
        """
        Create a new set of transform for each tile in the tiles dictionary
        :param tiles: Dictionary of tile ID to tiles
        :return: sorted list of transforms for ID's found tiles
        """

        transforms = []

        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue

            tile = tiles[ID]

            transform = self._CreateTransform(ID,
                                              full_res_image_shape=tile.ImageSize * tile.image_to_source_space_scale)

            tile.Transform = transform

        return transforms

    def ToMosaicTileset(self, tiles):
        """
        Creates a new MosaicTileset object.  Copies tiles from the provided MosaicTileset but with updated transforms.
        Output tileset is translated to zero origin

        :param dict tiles: Maps tile ID used in layout to a Tile object
        """

        if len(tiles) == 0:
            raise ValueError('tiles parameter expected to have at least one tile')

        first_tile_id = next(iter(tiles))
        image_to_source_space_scale = tiles[first_tile_id].image_to_source_space_scale

        mosaic_tileset = nornir_imageregistration.mosaic_tileset.MosaicTileset(
            image_to_source_space_scale=image_to_source_space_scale)

        for ID in sorted(tiles.keys()):
            if not ID in self.nodes:
                continue

            tile = copy.deepcopy(tiles[ID])
            tile.Transform = self._CreateTransform(ID,
                                                   full_res_image_shape=tile.ImageSize * tile.image_to_source_space_scale)
            mosaic_tileset[tile.ID] = tile

        mosaic_tileset = mosaic_tileset.TranslateToZeroOrigin()

        return mosaic_tileset

    def ToMosaic(self, tiles):
        """
        Generate a Mosaic object from a dictionary mapping tile numbers to tile paths that can be used to create a .mosaic file
        :param dict tiles: Maps tile ID used in layout to a Tile object
        """
        warnings.warn("Soon to be deprecated, use ToMosaicTileset instead")

        mosaic_tileset = self.ToMosaicTileset(tiles)
        return mosaic_tileset.ToMosaic()


def OffsetsSortedByWeight(layout: Layout) -> NDArray:
    """
    Return all of a layouts offsets sorted by weight.
    :return: An array [[TileA_ID, TileB_ID, OffsetY, OffsetX, Weight]] To prevent duplicates we only report offsets where TileA_ID < TileB_ID
    """
    ret_array = np.empty((0, 5))
    for node in layout.nodes.values():
        if node.IsIsolated:
            continue

        # Prevent duplicates by skipping IDs less than the nodes
        iNewRows = node.OffsetArray[:, 0] > node.ID
        if not np.any(iNewRows):
            continue

        new_column = np.ones((int(np.sum(iNewRows)), 1)) * node.ID
        new_rows = np.hstack((new_column, node.OffsetArray[iNewRows, :]))
        ret_array = np.vstack((ret_array, new_rows))

    return _sort_array_on_column(ret_array, 4)


def ScaleOffsetWeightsByPosition(original_layout):
    """Scale each node's offset weights by the positions of linked nodes.

    :raises NotImplementedError: Always, for any layout with at least one node.  The per-node
        method this delegates to was never implemented.  Unlike its three siblings in this
        module it has no working behaviour to preserve, and its only call site in
        arrange_mosaic is commented out.  See #250.
    """
    for node in original_layout.nodes.values():
        linked_node_positions = original_layout.GetNodes(node.ConnectedIDs)
        node.ScaleOffsetWeightsByPosition(linked_node_positions)

    return


def NormalizeOffsetWeights(original_layout: Layout,
                           min_allowed_weight: float | None = None,
                           max_allowed_weight: float | None = None) -> None:
    """Scale offset weights proportionally so they lie in [min_allowed_weight, max_allowed_weight].

    Modifies nodes in original_layout in place. Isolated nodes are skipped.

    :param original_layout: Layout whose node offset weights to normalize.
    :param min_allowed_weight: Minimum weight after scaling.  Defaults to 0.  Set equal to
        max_allowed_weight to give every offset the same weight.
    :param max_allowed_weight: Maximum weight after scaling.  Defaults to 1.
    :return: None.
    """

    # These name the *output* range, matching the docstring above, the wording in
    # TranslateSettings ("the minimum weight we will allow an offset measurement between two
    # tiles to have"), and how ScaleOffsetWeightsByPopulationRank reads the same two
    # parameters. They were previously assigned over minWeight/maxWeight, the *source* extrema,
    # so the output was always [0, 1] and the configured floor never reached a weight. Worse
    # than ignored: with min_allowed_weight=0.5 on weights 0.1-0.9, three of five links were
    # driven to exactly zero, losing their pull in the relaxation entirely, when the caller had
    # asked for nothing below 0.5 (#130).
    #
    # The defaults reproduce the old output for the default TranslateSettings, where both are
    # None, so only a configuration that sets them changes.
    min_allowed = 0.0 if min_allowed_weight is None else float(min_allowed_weight)
    max_allowed = 1.0 if max_allowed_weight is None else float(max_allowed_weight)

    if min_allowed > max_allowed:
        raise ValueError(
            f"min_allowed_weight ({min_allowed}) must not exceed max_allowed_weight "
            f"({max_allowed})")

    (minWeight, maxWeight) = original_layout.GetOffsetWeightExtrema()

    allowed_range = max_allowed - min_allowed
    weight_range = maxWeight - minWeight

    # Either the layout has nothing to spread, or the caller asked for a single weight by
    # passing min == max. Equality is a documented TranslateSettings option, so it is honoured
    # rather than rejected the way the population-rank helper rejects it.
    if np.isclose(maxWeight, minWeight) or np.isclose(allowed_range, 0):
        for node in original_layout.nodes.values():
            if node.IsIsolated:
                continue

            node.Weights = max_allowed

        return

    for node in original_layout.nodes.values():
        # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
        if node.IsIsolated:
            continue

        node.Weights = (((node.Weights - minWeight) / weight_range) * allowed_range) + min_allowed

        if nornir_imageregistration.in_debug_mode():
            assert (np.all(node.Weights >= min_allowed))
            assert (np.all(node.Weights <= max_allowed))
        else:
            node.Weights = np.clip(node.Weights, min_allowed, max_allowed)
    return


def SetOffsetWeights(original_layout: Layout, weight_value: float) -> None:
    """Set all non-isolated nodes' offset weights to weight_value. Modifies layout in place; returns None."""
    for node in original_layout.nodes.values():
        if node.IsIsolated:
            continue

        # Through the Weights setter: OffsetArray hands back a read-only copy (it says
        # "Read-only use please"), so assigning into it raised "ValueError: assignment
        # destination is read-only" and this function could never run at all (#130).
        node.Weights = weight_value


def ScaleOffsetWeightsByPopulationRank(original_layout: Layout,
                                       min_allowed_weight: float = 0,
                                       max_allowed_weight: float = 1.0) -> None:
    """Remap offset weights by population rank so they span [min_allowed_weight, max_allowed_weight].

    Modifies nodes in original_layout in place. Isolated nodes get max_allowed_weight when all weights are equal.

    :param original_layout: Layout whose node offset weights to scale.
    :param min_allowed_weight: Target minimum weight (must be < max_allowed_weight).
    :param max_allowed_weight: Target maximum weight.
    :return: None.
    """

    if min_allowed_weight >= max_allowed_weight:
        raise ValueError("Min allowed weight must be below the max allowed weight")

    (minWeight, maxWeight) = original_layout.GetOffsetWeightExtrema()

    # All the weights are equal... odd
    if maxWeight == minWeight:
        for node in original_layout.nodes.values():
            if node.IsIsolated:
                continue

            node.Weights = max_allowed_weight
        return

    # Workaround for all weights being pretty decent and therefore a weight is artificially considered bad
    maxWeight -= minWeight

    allowed_weight_range = max_allowed_weight - min_allowed_weight

    for node in original_layout.nodes.values():
        # Sometimes we have tiles which end up isolated, usually due to prune.  When this occurs they have no scores
        if node.IsIsolated:
            continue

        # Through the Weights setter, for the read-only reason noted in SetOffsetWeights.
        node.Weights = (((node.Weights - minWeight) / maxWeight) * allowed_weight_range) \
            + min_allowed_weight

        if nornir_imageregistration.in_debug_mode():
            assert (np.all(node.Weights >= min_allowed_weight))
            assert (np.all(node.Weights <= max_allowed_weight))
        else:
            node.Weights = np.clip(node.Weights, min_allowed_weight, max_allowed_weight)

    return


def RelaxLayout(layout_obj: Layout, max_tension_cutoff=None, max_iter=None, vector_scale=None, min_improvement=0.001,
                plotting_output_path=None, plotting_interval=None) -> Layout:
    """
    :param vector_scale:
    :param plotting_output_path:
    :param plotting_interval:
    :param layout_obj: Layout to refine
    :param float max_tension_cutoff: Stop iteration after the maximum tension vector has a magnitude below this value
    :param int max_iter: Maximum number of iterations
    :param float min_improvement: The max tension must decrease by at least this amount or the loop will exit
    """

    max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]

    if max_tension_cutoff is None:
        max_tension_cutoff = 0.1

    if max_iter is None:
        max_iter = 500

    if plotting_interval is None:
        plotting_interval = 10

    i = 0
    min_plotting_tension = max_tension_cutoff * 20
    # plotting_max_tension = max(min_plotting_tension, max_tension)

    #         MovieImageDir = os.path.join(self.TestOutputPath, "relax_movie")
    #         if not os.path.exists(MovieImageDir):
    #             os.makedirs(MovieImageDir)
    pool = None
    if plotting_output_path is not None:
        os.makedirs(plotting_output_path, exist_ok=True)
        pool = nornir_pools.GetGlobalMultithreadingPool()

    prettyoutput.Log("Relax Layout")

    last_max_tension = None  # Used to track mean
    num_in_avg = 10
    delta_avg = []
    delta_mean = 100000
    if min_improvement is not None:
        delta_mean = min_improvement + 1.0

        # last_output = ""
    while max_tension > max_tension_cutoff and i < max_iter:

        # Stop the loop if we aren't making good progress
        if min_improvement is not None and delta_mean < min_improvement:
            prettyoutput.Log(
                f'Min improvement threshold not met, delta was {delta_mean:.4f}, which is below {min_improvement:0.4f}, stopping.')
            break

        if i <= 10 or i % 25 == 0:
            prettyoutput.CurseProgress(f'Pass #{i:d} Max: {max_tension:0.4g}', i)
        # sys.stdout.write('\b' * len(last_output))
        # output_str = "\tPass #%d %g" % (i, max_tension)
        # sys.stdout.write(output_str)
        # sys.stdout.flush()
        # last_output = output_str
        Layout.RelaxNodes(layout_obj, vector_scalar=vector_scale)
        max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]

        if plotting_output_path is not None and (i % plotting_interval == 0 or i < plotting_interval):
            plotting_max_tension = max(min_plotting_tension, max_tension)
            filename = os.path.join(plotting_output_path, "%d.svg" % i)
            #             nornir_imageregistration.views.plot_layout(
            #                            layout_obj=layout_obj.copy(),
            #                            OutputFilename=filename,
            #                            max_tension=plotting_max_tension)
            layout_obj_copy = layout_obj.copy()
            layout_obj_copy.TranslateToZeroOrigin()
            assert pool is not None
            pool.add_task("Plot step #%d" % i,
                          nornir_imageregistration.views.plot_layout,
                          layout_obj=layout_obj_copy,
                          OutputFilename=filename,
                          max_tension=plotting_max_tension)

        # node_distance = setup_imagetest.array_distance(node_movement[:,1:3])             
        # max_distance = np.max(node_distance,0)

        # Update the running average of progress each iteration
        if last_max_tension is not None:
            delta = last_max_tension - max_tension
            delta_avg.append(delta)

            if len(delta_avg) > num_in_avg:
                del delta_avg[0]
                delta_mean = np.array(delta_avg).mean()

        last_max_tension = max_tension

        i += 1

        # nornir_shared.plot.VectorField(layout_obj.GetPositions(), layout_obj.NetTensionVectors(), OutputFilename=filename)
        # pool.add_task("Plot step #%d" % (i), nornir_shared.plot.VectorField,layout_obj.GetPositions(), layout_obj.WeightedNetTensionVectors(), OutputFilename=filename)

    prettyoutput.Log("\n")
    return layout_obj


def BuildLayoutWithHighestWeightsFirst(original_layout):
    """
    Constructs a mosaic by sorting all of the match results according to strength.
    :param original_layout: Dictionary of tile objects containing alignment records to other tiles
    """

    # placedTiles = dict()

    sorted_offsets = OffsetsSortedByWeight(original_layout)

    print("Building Layout from offsets")
    LayoutList = []
    for iRow in range(0, sorted_offsets.shape[0]):
        row = sorted_offsets[iRow, :]
        A_ID = int(row[0])
        B_ID = int(row[1])
        # YOffset = row[2]
        # XOffset = row[3]
        Weight = row[4]
        offset = row[2:4]

        # print("%d -> %d (%g,%g w: %g)" % (A_ID, B_ID, row[2], row[3], Weight))

        if np.isnan(Weight):
            print("Skip: Invalid weight, not a number")
            continue

        ALayout = GetLayoutForID(LayoutList, A_ID)
        BLayout = GetLayoutForID(LayoutList, B_ID)

        if ALayout is None and BLayout is None:
            new_layout = Layout()
            A_pos = original_layout.GetPosition(A_ID)
            new_layout.CreateNode(A_ID, A_pos)
            new_layout.CreateNode(B_ID, A_pos + offset)
            new_layout.SetOffset(A_ID, B_ID, offset, Weight)
            LayoutList.append(new_layout)
            # print("New layout")

        elif (not ALayout is None) and (not BLayout is None):
            # Need to merge the layouts? See if they are the same
            if ALayout == BLayout:
                # Already mapped
                if B_ID in ALayout.nodes[A_ID].ConnectedIDs:
                    # print("Skip: Already mapped")
                    pass
                else:
                    ALayout.SetOffset(A_ID, B_ID, offset, Weight)
            else:
                MergeLayoutsWithNodeOffset(ALayout, BLayout, A_ID, B_ID, offset, Weight)
                # print("Merged")
                LayoutList.remove(BLayout)
        else:

            if ALayout is None and not BLayout is None:
                BLayout.CreateOffsetNode(B_ID, A_ID, -offset, Weight)
                # We'll pick it up on the next pass
                # print("Skip: Getting it next time")
                # continue

            else:
                assert ALayout is not None
                ALayout.CreateOffsetNode(A_ID, B_ID, offset, Weight)

    # OK, we should have a single list of layouts
    # LargestLayout = LayoutList[0]

    return LayoutList


def MergeDisconnectedLayouts(layout_list: list[Layout]) -> Layout:
    """Given a list of layouts, generate a single layout with all nodes in the same positions."""
    if len(layout_list) == 1:
        return layout_list[0]

    # Find the nearest two tiles from both layouts. 
    # Create an artificial link between the tiles based on current coordinates
    # Then merge the nodes

    merged_layout = layout_list[0].copy()
    A = [(n.ID, n.Position) for n in merged_layout.nodes.values()]
    matrix_A = np.vstack([row[1] for row in A])

    for (i, other_layout) in enumerate(layout_list):
        if i == 0:
            continue

        B = [(n.ID, n.Position) for n in other_layout.nodes.values()]

        matrix_B = np.vstack([row[1] for row in B])

        # Tile-count pairwise (tens–hundreds), already NumPy. Not a CuVS path:
        # N is layouts, not control points, and data is on the host.
        distances = pairwise_cdist(matrix_A, matrix_B, metric='sqeuclidean')
        A_min = np.min(distances, 1)
        B_min = np.min(distances, 0)
        iA = np.argmin(A_min)
        iB = np.argmin(B_min)
        A_ID = A[iA][0]
        B_ID = B[iB][0]

        offset = matrix_B[iB, :] - matrix_A[iA, :]
        merged_layout.Merge(other_layout)
        merged_layout.SetOffset(A_ID, B_ID, offset, 1.0)

        A.extend(B)  # Add the entries in B for the next loop
        matrix_A = np.vstack((matrix_A, matrix_B))

    return merged_layout


def _generate_combinations(list_of_lists):
    """Yield all pairs (A, B) where A and B come from different lists and A < B.

    :param list_of_lists: List of iterables of integers (e.g. section IDs per group).
    :return: Iterator of (A, B) pairs.
    """
    for (iList, id_list) in enumerate(list_of_lists):
        for (iOther, other_list) in enumerate(list_of_lists):
            if iOther <= iList:
                continue

            for A in id_list:
                for B in other_list:
                    if A < B:
                        yield A, B


def MergeDisconnectedLayoutsWithOffsets(layout_list, tile_offset_dict=None):
    """Merge multiple layouts into one using optional pairwise tile offsets.

    :param layout_list: List of Layout objects to merge; returns None if empty/None.
    :param tile_offset_dict: Optional dict mapping (A, B) pair IDs to [Y, X] offsets.
    :return: Single merged Layout, or None if layout_list is empty/None or has one element (returns that element).
    """
    if layout_list is None or len(layout_list) == 0:
        return None

    if len(layout_list) == 1:
        return layout_list[0]

    if tile_offset_dict is None:
        tile_offset_dict = {}
    else:
        # Clone the dictionary so we don't change the passed parameter
        tile_offset_dict = dict(tile_offset_dict)

    # First, to minimize our search time, remove offsets the layouts already encode
    # and build a frozenset of IDs in the layouts
    layout_IDs = []
    tile_to_layout = {}  # Track which tile belongs to which layout
    for (iLayout, layout) in enumerate(layout_list):
        layout_IDs.append(frozenset([node_id for node_id in layout.nodes.keys()]))

        for link in layout.linked_nodes:
            assert (link[0] < link[1])
            if link in tile_offset_dict:
                del tile_offset_dict[link]

        for node_id in layout.nodes.keys():
            tile_to_layout[node_id] = iLayout

    # Remove the tile links who are in the same layout but not linked in the layout
    for key in list(tile_offset_dict.keys()):
        if key[0] in tile_to_layout and key[1] in tile_to_layout:
            if tile_to_layout[key[0]] == tile_to_layout[key[1]]:
                del tile_offset_dict[key]

    cross_layout_keys = {}

    # Identify all of the tile_offsets that can describe where the layouts are relative to each other
    for offset_key in tile_offset_dict.keys():
        try:
            iLayout_A = tile_to_layout[offset_key[0]]
            iLayout_B = tile_to_layout[offset_key[1]]
        except KeyError:
            continue

        assert (iLayout_A != iLayout_B)

        layout_pair = (iLayout_A, iLayout_B)
        if iLayout_B < iLayout_A:
            layout_pair = (iLayout_B, iLayout_A)

        if iLayout_A != iLayout_B:
            if layout_pair in cross_layout_keys:
                cross_layout_keys[layout_pair].append(offset_key)
            else:
                cross_layout_keys[layout_pair] = [offset_key]

    # Generate a list of the layouts pairings with the greatest number of keys first, merge largest to smallest
    layout_pair_offset_count = [(key, len(cross_layout_keys[key])) for key in cross_layout_keys.keys()]
    sorted_layout_pair_offset_count = sorted(layout_pair_offset_count, key=itemgetter(1), reverse=True)

    # layout_centers = np.vstack([l.average_center for l in layout_list])

    # Merge the largest layouts to the smallest layouts until all are merged that can be merged
    while len(sorted_layout_pair_offset_count) > 0:
        layout_pair_to_merge = sorted_layout_pair_offset_count.pop(0)
        layout_pair = layout_pair_to_merge[0]
        (iLayout_A, iLayout_B) = layout_pair
        assert (iLayout_A != iLayout_B)
        ALayout = layout_list[iLayout_A]
        BLayout = layout_list[iLayout_B]

        if ALayout is BLayout:
            # An earlier pair already merged these two. The index comparison above
            # cannot detect that: layout_list slots are rebound as layouts merge,
            # but the pending pair list and tile_to_layout keep the original
            # indices, so distinct indices no longer imply distinct layouts. Any
            # set of three pairwise-connected layouts produces such a pair.
            #
            # Merging here would hand the same object to MergeLayoutsWithAbsoluteOffset
            # as both A and B, translating the merged layout against itself by a
            # residual offset -- measured [-965.67, 972.33] on three layouts -- and
            # then merging it with itself, which is a no-op. The translation is
            # uniform, so relative geometry survives and the usual
            # TranslateToZeroOrigin afterwards hides it entirely; a caller that
            # skips that normalization sees a displaced mosaic instead.
            #
            # The offsets in this pair are redundant constraints on a merge that
            # already happened, so dropping them loses no information here; the
            # relaxation pass is what reconciles competing offsets.
            continue

        print("Layout {0} absorbing {1}".format(iLayout_A, iLayout_B))

        tile_offsets = cross_layout_keys[layout_pair]

        A_To_B_offset_measures = np.zeros((len(tile_offsets), 2))

        measured_count = 0
        for offset_key in tile_offsets:

            # Using each tile offset key, average the offset between the disconnected layouts
            tile_offset = tile_offset_dict[offset_key]
            iLayout = (tile_to_layout[offset_key[0]], tile_to_layout[offset_key[1]])
            # assert(iLayout[0] != iLayout[1])
            if iLayout[0] == iLayout[1]:
                continue

            tile_layouts = (layout_list[iLayout[0]], layout_list[iLayout[1]])
            A_Pos = tile_layouts[0].GetPosition(offset_key[0])
            B_Pos = tile_layouts[1].GetPosition(offset_key[1])
            # Layout_A_Center = layout_centers[iLayout[0], :]
            # Layout_B_Center = layout_centers[iLayout[1], :]
            # A_To_Layout = A_Pos - Layout_A_Center
            # B_To_Layout = B_Pos - Layout_B_Center
            # A_To_B = A_To_Layout + tile_offset + B_To_Layout

            # A_To_B_offset_measures[iRow, :] = A_To_B
            A_To_B_offset_measures[measured_count, :] = (B_Pos - A_Pos) - tile_offset
            measured_count += 1

            # MergeLayoutsWithNodeOffset(ALayout, BLayout, offset_key[0], offset_key[1], tile_offset.offset, Weight=0)
            # print("Merged")

        # Average only the rows actually measured. The skip above cannot fire today,
        # because cross_layout_keys is built solely from keys whose layout indices
        # differ (same-layout keys are deleted above, and insertion is guarded), and
        # tile_to_layout is never updated as layouts merge, so this recomputes the
        # same unequal indices. If a future change arms that guard, a leftover zero
        # row would drag the merge offset toward the origin by measured/total --
        # 0.6x for 2 skipped of 5 -- silently displacing every tile in the absorbed
        # layout rather than failing.
        assert measured_count > 0, \
            f'No cross-layout offsets measured for layout pair {layout_pair}'
        A_To_B_offset_measures = A_To_B_offset_measures[:measured_count]

        MergeLayoutsWithAbsoluteOffset(ALayout, BLayout, np.mean(A_To_B_offset_measures, axis=0))

        for (iLayout, layout) in enumerate(layout_list):
            if layout.ID == BLayout.ID:
                layout_list[iLayout] = ALayout

        # layout_list[iLayout_B] = ALayout
        # layout_centers[iLayout_A] = ALayout.average_center
        # layout_centers[iLayout_B] = ALayout.average_center

    # Now we check for layouts that are completely disconnected
    distinct_IDs = set([ll.ID for ll in layout_list])
    unmerged_layouts = []
    for layout in layout_list:
        if layout.ID in distinct_IDs:
            unmerged_layouts.append(layout)
            distinct_IDs -= {layout.ID}

    return MergeDisconnectedLayouts(unmerged_layouts)


def GetLayoutForID(listLayouts: list[Layout], ID: int) -> Layout | None:
    """Given a list of tile layouts, returns the layout containing the given ID."""

    if listLayouts is None:
        return None

    for layout in listLayouts:
        if layout.Contains(ID):
            return layout

    return None


def MergeLayoutsWithNodeOffset(layoutA, layoutB, NodeInA, NodeInB, offset, weight):
    """
    Merge B with A by translating all B transforms by offset.
    Then update the dictionary of A
    """

    PositionInA = layoutA.GetPosition(NodeInA)
    PositionInB = layoutB.GetPosition(NodeInB)

    ExpectedMovingTilePosition = offset + PositionInA
    MovingPositionDifference = ExpectedMovingTilePosition - PositionInB

    layoutB.Translate(MovingPositionDifference)
    layoutA.Merge(layoutB)

    layoutA.SetOffset(NodeInA, NodeInB, offset, weight)


def MergeLayoutsWithAbsoluteOffset(layoutA: Layout, layoutB: Layout, offset: NDArray | tuple[float, float]) -> None:
    """
    Merge B with A by translating all B transforms by offset.
    Then update the dictionary of A
    """

    layoutB.Translate(offset)
    layoutA.Merge(layoutB)

