
1.1.8
-----

**New**

* Added bounding box to spatial
* Python 3 support

1.1.7
-----

**New**

* RegistrationTree supports missing centers by finding the nearest section to the requested center

1.1.3
-----

**New**

* Add flag to StosBrute to allow execution on cluster
* AssembleTiles method allows specifying a subregion to assemble
* ShowGrayscale function displays multiple images on a 2D grid for a more optimal use of screen real estate
* Added Spatial package which includes enums for standard indexing of nornir spatial arrays.  For example iArea.width is the index to use to obtain the width from a size tuple

**Changed**

* Bounds functions now follow the (MinY MinX MaxY MaxX) convention consistent with numpy image array indexing
* Alignment record now uses the standard (Y,X) indexing of the other image_registration packages.
* Removed many uses of ImageMagick identify to obtain image size and replaced with Pillow calls

**Fixed** 

* Image padding was making images larger than they had to be.  This fix should increase registration speed

1.1.0
-----

* Initial release