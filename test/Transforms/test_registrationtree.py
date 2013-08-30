'''
Created on Apr 25, 2013

@author: u0490822
'''
import unittest

from nornir_imageregistration.transforms import registrationtree

def checkRegistrationTree(test, RT, ControlSection, MappedSectionList):
    Node = RT.Nodes[ControlSection]

    MappedSectionList.sort()
    test.assertTrue(Node.Children == MappedSectionList)
    return


class TestRegistrationTree(unittest.TestCase):

    def testName(self):
        '''Generate registration trees and check the contents'''

        center = 1
        numbers = range(1, 5)
        RT = registrationtree.RegistrationTree.CreateRegistrationTree(numbers, adjacentThreshold=2, center=center)
        self.assertTrue(center in RT.RootNodes, "Center number not in RootNodes")

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


        numbers = range(1, 6)
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



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()