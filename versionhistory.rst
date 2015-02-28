
1.2.4
-----

**Fixed**

* Support for translating tiles in a mosaic based on stage position
* Bug fixes

1.2.3
-----

**New**

* Support JPEG2000 and memory mapped numpy arrays as image output formats
* Use a generator to iterate through tiles pulled from a larger image.
* Some significant memory footprint reductions for assemble and transformation functions 

**Fixed**

* Do not throw an exception when asking a .stos file for the full path to a mask that it doesn not have.  Return None instead.


1.2.2
-----

Minor optimization and function used in the new tile web server

**New**

* Added a function to cut image into tiles
* Added a resize image function
* Reduced memory footprint of assemble somewhat.


1.2.1
-----

**New**

* Many optimizations to assembling images
* Tests will be profiled if the PROFILE environment variable is set

1.2.0
-----

**New**

* Added bounding box to spatial
* Python 3 support
* Better documentation for assemble parameters.
* Added bounding box structure to spatial module

**Fixed**

* The output buffer in assemble is now the correct size.  Previously a larger buffer was allocated and cropped.

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