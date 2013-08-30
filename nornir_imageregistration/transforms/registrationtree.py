'''
Created on Nov 14, 2012

@author: u0490822
'''

class RegistrationTreeNode(object):

    @property
    def IsRoot(self):
        return self.Parent is None;

    def __init__(self, sectionNumber):
        self.Parent = None;
        self.SectionNumber = sectionNumber;
        self.Children = [];

    def SetParent(self, sectionNumber):
        self.Parent = sectionNumber;

    def AddChild(self, childSectionNumber):
        self.Children.append(childSectionNumber);
        self.Children.sort()

    def __str__(self, *args, **kwargs):
        s = str(self.SectionNumber) + " <- "
        for c in self.Children:
            s += str(c) + ' '

        return s



class RegistrationTree(object):
    '''
    The registration tree tracks which sections are mapped to each other.  When calculating the section to volume transform we begin with the
    root nodes and register down the tree until registration is completef
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.Nodes = {}  # Dictionary of all nodes
        self.RootNodes = {};  # Dictionary of nodes without parents

    def AddPair(self, ControlSection, MappedSection):

        if ControlSection == MappedSection:
            return

        ControlNode = None;
        if ControlSection in self.Nodes:
            ControlNode = self.Nodes[ControlSection]
        else:
            ControlNode = RegistrationTreeNode(ControlSection);
            self.Nodes[ControlSection] = ControlNode;
            self.RootNodes[ControlSection] = ControlNode;

        ControlNode.AddChild(MappedSection);

        if MappedSection in self.Nodes:
            MappedNode = self.Nodes[MappedSection]

            # Remove mapped node from the root node
            if MappedSection in self.RootNodes:
                del self.RootNodes[MappedSection];
        else:
            MappedNode = RegistrationTreeNode(MappedSection);
            self.Nodes[MappedSection] = MappedNode;

        MappedNode = self.Nodes.get(MappedSection, RegistrationTreeNode(ControlSection));


    @classmethod
    def CreateRegistrationTree(cls, sectionNumbers, adjacentThreshold=2, center=None):
        sectionNumbers.sort()
        RT = RegistrationTree()

        centerindex = (len(sectionNumbers) - 1) / 2
        if not center is None:
            centerindex = sectionNumbers.index(center)

        listAdjacentBelowCenter = AdjacentPairs(sectionNumbers, adjacentThreshold, startindex=0, endindex=centerindex)
        listAdjacentAboveCenter = AdjacentPairs(sectionNumbers, adjacentThreshold, startindex=len(sectionNumbers) - 1, endindex=centerindex)

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


def AdjacentPairs(sectionNumbers, adjacentThreshold, startindex, endindex):


    listAdjacent = []
    if startindex == endindex:
        return listAdjacent

    step = 1
    if startindex > endindex:
        step = -1

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