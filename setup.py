'''
Created on Aug 30, 2013

@author: u0490822
'''

import glob
import os

from ez_setup import use_setuptools

# from setuptools import setup, find_packages
if __name__ == '__main__':
    use_setuptools()

    from setuptools import setup, find_packages

    packages = find_packages()
    
    #OK to use pools v1.3.1, no changes made for v1.3.2
    
    #Starting with 1.3.4 Image Magick 7 is required

    install_requires = ["nornir_pools>=1.4",
                        "nornir_shared>=1.4",
                        "numpy>=1.9.1",
                        "scipy>=0.13.2",
                        "matplotlib>=1.3.0",
                        "Pillow-SIMD>=5.3",
                        "six"]

    dependency_links = ["git+https://github.com/nornir/nornir-pools#egg=nornir_pools-1.4",
                        "git+https://github.com/nornir/nornir-shared#egg=nornir_shared-1.4"]

    scripts = ['nornir-addtransforms = nornir_imageregistration.scripts.nornir_addtransforms:Execute',
               'nornir-assemble-tiles = nornir_imageregistration.scripts.nornir_assemble_tiles:Execute',
               'nornir-assemble = nornir_imageregistration.scripts.nornir_assemble:Execute',
               'nornir-rotate-transalate = nornir_imageregistration.scripts.nornir_rotate_translate:Execute',
               'nornir-slice-to-mosaic = nornir_imageregistration.scripts.nornir_slicetomosaic:Execute',
               'nornir-translatemosaic = nornir_imageregistration.scripts.nornir_translatemosaic:Execute',
               'nornir-scaletransform = nornir_imageregistration.scripts.nornir_scaletransform:Execute',
               'nornir-stos-grid-refinement = nornir_imageregistration.scripts.nornir_stos_grid_refinement:Execute'
               'nornir-show-mosaic-layout = nornir_imageregistration.scripts.nornir_show_mosaic_layout:Execute'
               ]
               
    
    # named_scripts = []
    
    # script_template = '%s = %s'
    # for script_path in scripts:
        # renamed = get_script_name(script_path)
        # entry = script_template % (renamed, script_path)
        # named_scripts.append(entry)
        
    entry_points = {'console_scripts' : scripts}

    classifiers = ['Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering']

    setup(name='nornir_imageregistration',
          zip_safe=False,
          classifiers=classifiers,
          version='1.4.0',
          description="Contains the core image registration algorithms for aligning 2d images into larger mosaics and 3D volumes",
          author="James Anderson",
          author_email="James.R.Anderson@utah.edu",
          url="https://github.com/nornir/nornir-imageregistration",
          packages=packages,
          entry_points=entry_points,
          test_suite='test',
          install_requires=install_requires,
          dependency_links=dependency_links)
