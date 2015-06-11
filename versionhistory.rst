
1.3.0
-----

**New**

* The original ir-refine-translate code from SCI has been replaced with a Python implementation.  This appears to solve problems where tiles in mosaics were occasionally out of position
* Richer API for Rectangles.  Added functions to change the area but keep the same center, safe rounding to an integer that does not decrease the overall area, 
* Added a crude measure (Difference of overlapping regions) of mosaic quality
* One can include titles in the ShowGrayscale function now

**Fixed**

* Close pools more agressively.  The extra threads multiprocessing pools spawn have a measurable performance cost, at least when running in low priority mode
* Do not crash when a file is missing if we are checking a list of tiles and transforms for the most common scalar.
* Adding stos transforms no longer removes masks from the .stos file.
* .stos file object's mask FullPath functions no longer crash if passed None.  Instead masks are properly cleared now.
* .stos file can read mask paths correctly from .stos files, was reading the header line as a mask name before.
* If all pixels are masked then do not try to calculate statistics for unmasked pixels.
* Fixed mask generation in RandomNoiseMask
* Fixed crash when MostCommonScalar function encounters a missing image file
* AddStosTransforms was not passing masks along to the new .stos transform
* We now raise a value error when a random noise mask does contain any unmasked pixels to generate statistics from.
* Fixed output of unmapped pixels when warped image call is mapping a single image and not a list of images.
* Fixed script generation on install.  Scripts for individual operations such as assemble should appear now.  More testing required here.
* Fixes for the use of memory mapped files.  These are still disabled in production.
* Fix for crop image when the cropped image is entirely outside the boundaries of the input array.


**Changed**

* Show grayscale layout and title improvements
* ExtractRegion removed and replaced with faster, simpler CropImage function.  Now raises DeprecationWarning.
* Deleted unused functions
* Rectangle object now always stores bounds internally as a numpy array 
* OnTransformChanged does not fire on a thread unless there are multiple listeners.  Performance improvement
* Numpy floating point issues not raise an exception instead of a warning message.  Now using: numpy.seterr(all='raise') 

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