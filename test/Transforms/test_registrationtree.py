'''
Created on Apr 25, 2013

@author: u0490822
'''
import unittest

from nornir_imageregistration.transforms import registrationtree


def _NodesToNumberList(Nodes):
    nums = []
    for n in Nodes:
        nums.append(n.SectionNumber)

    return nums

def checkRegistrationTree(test, RT, ControlSection, MappedSectionList):
    Node = RT.Nodes[ControlSection]

    MappedSectionList.sort()
    test.assertTrue(_NodesToNumberList(Node.Children) == MappedSectionList, "Expected %s to be %s" % (_NodesToNumberList(Node.Children), MappedSectionList))
    return


class TestRegistrationTree(unittest.TestCase):

    def testName(self):
        '''Generate registration trees and check the contents'''

        center = 1
        numbers = list(range(1, 5))
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=2, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        self.assertFalse(RT.IsEmpty, "Tree with nodes should not report itself as empty")

        checkRegistrationTree(self, RT, 1, [2, 3])
        checkRegistrationTree(self, RT, 2, [3, 4])
        checkRegistrationTree(self, RT, 3, [4])
        checkRegistrationTree(self, RT, 4, [])

        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        checkRegistrationTree(self, RT, 1, [2])
        checkRegistrationTree(self, RT, 2, [3])
        checkRegistrationTree(self, RT, 3, [4])
        checkRegistrationTree(self, RT, 4, [])

        center = 4
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=2, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        checkRegistrationTree(self, RT, 1, [])
        checkRegistrationTree(self, RT, 2, [1])
        checkRegistrationTree(self, RT, 3, [2, 1])
        checkRegistrationTree(self, RT, 4, [2, 3])


        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        checkRegistrationTree(self, RT, 1, [])
        checkRegistrationTree(self, RT, 2, [1])
        checkRegistrationTree(self, RT, 3, [2])
        checkRegistrationTree(self, RT, 4, [3])


        numbers = list(range(1, 6))
        center = 3
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=2, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        checkRegistrationTree(self, RT, 1, [])
        checkRegistrationTree(self, RT, 2, [1])
        checkRegistrationTree(self, RT, 3, [1, 2, 4, 5])
        checkRegistrationTree(self, RT, 4, [5])
        checkRegistrationTree(self, RT, 5, [])


        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

        checkRegistrationTree(self, RT, 1, [])
        checkRegistrationTree(self, RT, 2, [1])
        checkRegistrationTree(self, RT, 3, [2, 4])
        checkRegistrationTree(self, RT, 4, [5])
        checkRegistrationTree(self, RT, 5, [])



    def test_InvalidCenter(self):
        ''' Check registration tree using center that does not exist '''

        numbers = list(range(1, 3))
        numbers.extend(list(range(4, 7)))
        center = 3
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)

        self.assertEqual(len(RT.RootNodes), 1, "Should only be one root if center is missing")
        root = list(RT.RootNodes.values())[0]
        self.assertEqual(abs(root.SectionNumber - center), 1, "Adjusted center should be next to the specified center")

        checkRegistrationTree(self, RT, 1, [])
        checkRegistrationTree(self, RT, 2, [1, 4])
        checkRegistrationTree(self, RT, 4, [5])
        checkRegistrationTree(self, RT, 5, [6])
        checkRegistrationTree(self, RT, 6, [])


    def test_EmptyList(self):
        ''' Check registration tree using center that does not exist '''

        numbers = []
        center = 3
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        self.assertEqual(len(RT.RootNodes), 0, "Should not be a root if input was empty list")
        self.assertTrue(RT.IsEmpty, "Tree without nodes should report itself as empty")


    def test_SingleEntryList(self):
        ''' Check registration tree using center that does not exist '''

        numbers = [3]
        center = 2
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        self.assertEqual(len(RT.RootNodes), 1, "Should only be one root if center is missing")
        root = list(RT.RootNodes.values())[0]
        self.assertEqual(root.SectionNumber, numbers[0], "Root should be the only entry in the list")

        checkRegistrationTree(self, RT, 3, [])


    def test_InsertLeafOnlyNodes(self):
        ''' Check registration tree using center that does not exist '''

        numbers = [4, 7, 10, 12, 20, 30]
        center = 19
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=1, center=center)
        root = list(RT.RootNodes.values())[0]
        self.assertEqual(root.SectionNumber, 20, "Root should be closest node to requested value")

        checkRegistrationTree(self, RT, 4, [])
        checkRegistrationTree(self, RT, 7, [4])
        checkRegistrationTree(self, RT, 10, [7])
        checkRegistrationTree(self, RT, 12, [10])
        checkRegistrationTree(self, RT, 20, [12, 30])
        checkRegistrationTree(self, RT, 30, [])

        # Insert to leftmost possible node in the tree
        RT.AddNonControlSections([2, 5])
        checkRegistrationTree(self, RT, 4, [2])
        checkRegistrationTree(self, RT, 7, [4, 5])

        # Insert to the root of the tree
        RT.AddNonControlSections([13, 25])
        checkRegistrationTree(self, RT, 20, [12, 13, 25, 30])

        # Insert to rightmost possible node in the tree
        RT.AddNonControlSections([29, 31])
        checkRegistrationTree(self, RT, 20, [12, 13, 25, 29, 30])
        checkRegistrationTree(self, RT, 30, [31])

        # Make sure we cannot insert into a leaf node
        RT.AddNonControlSections([1, 32])
        checkRegistrationTree(self, RT, 4, [1, 2])
        checkRegistrationTree(self, RT, 30, [31, 32])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
