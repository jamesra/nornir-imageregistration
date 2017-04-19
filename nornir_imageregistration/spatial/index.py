'''
Created on Feb 28, 2014

@author: u0490822
'''

# #import rtree.index
#
#
# def CreateSpatialMap(boundingBoxes, objects):
#     '''
#     Given a set of bounding boxes, create an rtree index pointing at objects
#
#     :param list boundingBoxes: list of N tuples containing (MinX, MinY, MaxX, MaxY) bounding boxes
#     :param list objects: list of N objects to associate with bounding boxes.  Must be same length as boundingBoxes list or None
#
#     rtree pickles objects inserted into the tree.  This means we cannot modify returned objects, query them later, and get the modified object.
#     If you want that behavior store a key in the spatial index and use that key with an external dictionary or list of objects
#     '''
#
#     idx = rtree.index.Index()
#
#     for i, bb in enumerate(boundingBoxes):
#
#         obj = None
#         if objects:
#             obj = objects[i]
#
#         idx.insert(i, boundingBoxes[i], obj=obj)
#
#     return idx
#
# if __name__ == '__main__':
#     pass
