"""
Created on Nov 14, 2012

@author: u0490822
"""
from __future__ import annotations
import operator
from collections import namedtuple
from collections.abc import Iterable
import typing
from typing import Sequence, AbstractSet, Generator

MappedToRootWalkTuple = namedtuple('MappedToRootWalkTuple', ['RootNode', 'ParentNode', 'MappedNode'])


class RegistrationTreeNode(object):

    # @property
    # def IsRoot(self):
    #    return self.Parent is None

    @property
    def LeafOnly(self) -> bool:
        """Return true if this node must stay a leaf"""
        return self._leaf_only

    @property
    def NonLeafOnlyChildren(self) -> list[RegistrationTreeNode]:
        """Child nodes which can have child nodes"""
        output = []
        for c in self.Children:
            if not c.LeafOnly:
                output.append(c)

        return output

    @property
    def ChildSectionNumbers(self) -> Generator[int]:
        return (c.SectionNumber for c in self.Children)

    def __init__(self, sectionNumber: int, leaf_only: bool = False):
        # self.Parent = None
        self.SectionNumber = sectionNumber
        self.Children = []  # type: list[RegistrationTreeNode]
        self._leaf_only = leaf_only

    # def SetParent(self, sectionNumber):
    #    self.Parent = sectionNumber

    def AddChild(self, childSectionNumber: int):
        self.Children.append(childSectionNumber)
        self.Children.sort(key=operator.attrgetter('SectionNumber'))

    def __repr__(self):
        s = str(self.SectionNumber) + " <- "
        for c in self.Children:
            if c.LeafOnly:
                s += '[' + str(c.SectionNumber) + '] '
            else:
                s += str(c.SectionNumber) + ' '

        return s

    def FindControlForMapped(self,
                             mappedsection: int,
                             center: int | None = None) -> RegistrationTreeNode:
        """ Returns the best control section for the mapped section
            :param RegistrationTreeNode self: Root node to begin search on
            :param int mappedsection: Section number to locate a control node for
            :param center: Number of the root for the tree we are searching
        """
        if center is None:
            center = self.SectionNumber

        direction = mappedsection - center

        nodes_to_check = [self]

        while True:
            node_to_check = nodes_to_check.pop(0)

            children = node_to_check.NonLeafOnlyChildren

            if len(children) == 0:
                return node_to_check
            else:
                # Registration trees are setup so every child is either greater or less than our nodes value, except on the root node

                if direction < 0:
                    # Left insertion
                    for leftchild in children:
                        # Edge case, the root may have children greater than our value.
                        if leftchild.SectionNumber > node_to_check.SectionNumber:
                            continue

                        if mappedsection < leftchild.SectionNumber:
                            nodes_to_check.append(
                                leftchild)  # This is the largest step we can make towards the center section
                            break
                            # return self.FindControlForMapped(leftchild, mappedsection, center)

                    if len(nodes_to_check) == 0:
                        return node_to_check
                else:
                    # Right insertion
                    children.reverse()

                    for rightchild in children:
                        # Edge case, the root may have children greater than our value
                        if rightchild.SectionNumber < node_to_check.SectionNumber:
                            continue

                        if mappedsection > rightchild.SectionNumber:
                            nodes_to_check.append(
                                rightchild)  # This is the largest step we can make towards the center section
                            break
                            # return self.FindControlForMapped(rightchild, mappedsection, center)

                # No candidates to check, so rtnode is closer than all children and we should insert to ourselves
                if len(nodes_to_check) == 0:
                    return node_to_check

    def FindControlForMappedRecursive(self,
                                      mappedsection: int,
                                      center: int | None = None) -> RegistrationTreeNode:
        """ Returns the best control section for the mapped section
            :param RegistrationTreeNode self: Root node to begin search on
            :param int mappedsection: Section number to locate a control node for
            :param int center: Center for the tree we are searching
        """
        rtnode = self

        if center is None:
            center = rtnode.SectionNumber

        direction = mappedsection - center
        children = rtnode.NonLeafOnlyChildren

        if len(children) == 0:
            return rtnode
        else:
            # Registration trees are setup so every child is either greater or less than our nodes value, except on the root node

            if direction < 0:
                # Left insertion
                for leftchild in children:
                    # Edge case, the root may have children greater than our value
                    if leftchild.SectionNumber > rtnode.SectionNumber:
                        continue

                    if mappedsection < leftchild.SectionNumber:
                        return leftchild.FindControlForMapped(mappedsection, center)

                # Made it this far, so it is smaller than all children and we should insert to ourselves
                return rtnode
            else:
                # Right insertion
                children.reverse()

                for rightchild in children:
                    # Edge case, the root may have children greater than our value
                    if rightchild.SectionNumber < rtnode.SectionNumber:
                        continue

                    if mappedsection > rightchild.SectionNumber:
                        return rightchild.FindControlForMapped(mappedsection, center)

                # Made it this far, so it is smaller than all children and we should insert to ourselves
                return rtnode


class RegistrationTree(object):
    """
    The registration tree tracks which sections are mapped to each other.  When calculating the section to volume transform we begin with the
    root nodes and register down the tree until registration is completed
    """

    @property
    def SectionNumbers(self) -> list[int]:
        """List of all section numbers contained in tree"""
        return list(self.Nodes.keys())

    @property
    def IsEmpty(self) -> bool:
        return len(self.Nodes) == 0

    def __init__(self):
        """
        Constructor
        """
        # Dictionary of all nodes
        self.Nodes = {}  # type: dict[int, RegistrationTreeNode]
        # Dictionary of nodes without parents
        self.RootNodes = {}  # type: dict[int, RegistrationTreeNode]

    def __str__(self):
        s = "Roots: "
        for r in self.RootNodes:
            s += " " + str(r)

        s += '\r\nSections:'
        s += ','.join([str(s) for s in self.SectionNumbers])
        return s

    def _GetOrCreateRootNode(self, ControlSection) -> RegistrationTreeNode:
        ControlNode = None
        if ControlSection in self.Nodes:
            ControlNode = self.Nodes[ControlSection]
        else:
            ControlNode = RegistrationTreeNode(ControlSection)
            self.Nodes[ControlSection] = ControlNode
            self.RootNodes[ControlSection] = ControlNode

        return ControlNode

    def _GetOrCreateMappedNode(self, MappedSection, leaf_only=False) -> RegistrationTreeNode:
        MappedNode = None
        if MappedSection in self.Nodes:
            MappedNode = self.Nodes[MappedSection]
        else:
            MappedNode = RegistrationTreeNode(MappedSection, leaf_only=leaf_only)
            self.Nodes[MappedSection] = MappedNode

        return MappedNode

    def AddPair(self, ControlSection: int, MappedSection: int) -> RegistrationTreeNode | None:
        """
        Maps a section to a control section
        :param int ControlSection: Section to be used as a reference during registration
        :param int MappedSection: Section to be warped during registration
        :rtype: RegistrationTreeNode
        """

        if ControlSection == MappedSection:
            return

        ControlNode = self._GetOrCreateRootNode(ControlSection)
        MappedNode = self._GetOrCreateMappedNode(MappedSection)

        ControlNode.AddChild(MappedNode)
        # Remove mapped node from the root node
        if MappedSection in self.RootNodes:
            del self.RootNodes[MappedSection]

        return MappedNode

    def AddEmptyRoot(self, ControlSection) -> RegistrationTreeNode:
        return self._GetOrCreateRootNode(ControlSection)

    #     def NearestNode(self, sectionNumber, excludeLeafOnlyNodes=True):
    #         '''Return the node nearest to the requested sectionNumber'''
    #
    #         if sectionNumber in self.Nodes:
    #             return self.Nodes[sectionNumber]

    def _InsertLeafNode(self, parent: RegistrationTreeNode, sectionnum: int):
        """
            :param RegistrationTreeNode parent: Root node to consider insertion on
            :param int sectionnum: Section number to insert
        """

        leafonlynode = self._GetOrCreateMappedNode(sectionnum, leaf_only=True)
        parent.AddChild(leafonlynode)
        if leafonlynode.SectionNumber in self.RootNodes:
            raise ValueError("Leaf node cannot have the same number as an existing root node")

        return

    def AddNonControlSections(self, sectionNumbers: Iterable[int], center: int | None = None):
        """Add section numbers which cannot have sections registered to them, they exist as leaf nodes only"""

        assert (len(self.RootNodes) == 1)  # This should be used on trees that have only one root

        rtnode = list(self.RootNodes.values())[0]
        if center:
            center = NearestSection(self.SectionNumbers, center)
            rtnode = self.Nodes[center]

        for sectionNumber in sectionNumbers:
            parent = rtnode.FindControlForMapped(sectionNumber)
            self._InsertLeafNode(parent, sectionNumber)

        return

    @classmethod
    def CreateRegistrationTree(cls, sectionNumbers, adjacentThreshold=2, center=None) -> RegistrationTree:
        sectionNumbers = sorted(sectionNumbers)

        RT = RegistrationTree()

        if len(sectionNumbers) == 0:
            return RT
        elif len(sectionNumbers) == 1:
            RT.AddEmptyRoot(sectionNumbers[0])
            return RT
        else:
            centerindex = (len(sectionNumbers) - 1) // 2
            if center is not None:
                center = NearestSection(sectionNumbers, center)
                centerindex = sectionNumbers.index(center)

            listAdjacentBelowCenter = AdjacentPairs(sectionNumbers, adjacentThreshold, startindex=0,
                                                    endindex=centerindex)
            listAdjacentAboveCenter = AdjacentPairs(sectionNumbers, adjacentThreshold,
                                                    startindex=len(sectionNumbers) - 1, endindex=centerindex)

            for (mappedSection, controlSection) in listAdjacentBelowCenter + listAdjacentAboveCenter:
                RT.AddPair(ControlSection=controlSection, MappedSection=mappedSection)

            #        for imapped in range(0, centerindex):
            #            mappedSection = sectionNumbers[imapped]
            #
            #            for icontrol in range(imapped, centerindex):
            #                controlSection = sectionNumbers[icontrol]
            #                if (icontrol - imapped) < numadjacent:
            #                    RT.AddPair(ControlSection=controlSection, MappedSection=mappedSection)
            #                else:
            #                    break
            #
            #        for imapped in range(centerindex + 1, len(sectionNumbers)):
            #            mappedSection = sectionNumbers[imapped]
            #
            #            for icontrol in range(centerindex + 1, imapped):
            #                controlSection = sectionNumbers[icontrol]
            #                if (icontrol - imapped) < numadjacent:
            #                    RT.AddPair(ControlSection=controlSection, MappedSection=mappedSection)
            #                else:
            #                    break

            return RT

    def GenerateOrderedMappingsToRoots(self) -> (RegistrationTreeNode, RegistrationTreeNode):
        """
        Yields mappings to all root nodes in root -> leaf order.
        For any given mapped section N the root and any intermediate section
        mappings are returned before N
        :returns: (RootNode, MappedNode)
        """

        for root in self.RootNodes.values():
            yield from self.GenerateOrderedMappingsToRootNode(root)

    def GenerateOrderedMappingsToRootNode(self, rootNode: RegistrationTreeNode) -> (RegistrationTreeNode, RegistrationTreeNode, RegistrationTreeNode):
        """
        Yields mappings to control sections in root -> leaf order.
        So that for any given mapped section N the root and any intermediate section
        mappings are returned before N
        :returns: (RootNode, ParentNode, MappedNode) The root of the tree, parent node of the mapped section, and the mapped section
        """

        nodes_to_walk = [rootNode]
        alreadyMapped = set()

        while nodes_to_walk:
            mappedSectionNumber = nodes_to_walk.pop()

            if isinstance(mappedSectionNumber, RegistrationTreeNode):
                rtNode = mappedSectionNumber
                mappedSectionNumber = mappedSectionNumber.SectionNumber
            elif mappedSectionNumber in self.Nodes:
                rtNode = self.Nodes[mappedSectionNumber]
            else:
                raise ValueError("Unexpected mappedSectionNumber {0}".format(mappedSectionNumber))
                continue  # Not sure how we could reach this state

            alreadyMapped.union([mappedSectionNumber])

            for mapped in rtNode.Children:
                yield MappedToRootWalkTuple(rootNode, rtNode, mapped)
                if mapped.SectionNumber in self.Nodes and mapped.SectionNumber not in alreadyMapped:
                    nodes_to_walk.append(mapped.SectionNumber)


def NearestSection(sectionNumbers: Sequence[int], reqnumber: int) -> int | None:
    """Returns the section number nearest to the section number, or the same section number if the section exists, or None if the section list is empty"""
    if reqnumber in sectionNumbers:
        return reqnumber
    else:
        if len(sectionNumbers) == 1:
            return sectionNumbers[0]

        foundNumber = None

        maxSectionNumber = max(sectionNumbers)
        nearest = maxSectionNumber + reqnumber  # Just setting a value that is well out of range
        for s in sectionNumbers:
            dist = abs(reqnumber - s)
            if dist < nearest:
                foundNumber = s
                nearest = dist

        return foundNumber


def AdjacentPairs(sectionNumbers, adjacentThreshold, startindex, endindex):
    listAdjacent = []
    if startindex == endindex:
        return listAdjacent

    step = 1
    if startindex > endindex:
        step = -1

    assert (isinstance(startindex, int))
    assert (isinstance(endindex, int))
    assert (isinstance(step, int))

    for imapped in range(startindex, endindex + step, step):
        mappedSection = sectionNumbers[imapped]

        for icontrol in range(imapped + step, endindex + step, step):
            controlSection = sectionNumbers[icontrol]
            if (abs(icontrol - imapped)) <= adjacentThreshold:
                listAdjacent.append((mappedSection, controlSection))
                # RT.AddPair(ControlSection=controlSection, MappedSection=mappedSection)
            else:
                break

    return listAdjacent
