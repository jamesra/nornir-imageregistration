'''
Created on Aug 30, 2013

@author: u0490822
'''


from distutils.core import setup

required_packages = ["nornir_pools",
                     "nornir_shared",
                    "numpy",
                    "scipy",
                    "matplotlib"]

packages = ["nornir_imageregistration",
            "nornir_imageregistration.io",
            "nornir_imageregistration.geometry",
            "nornir_imageregistration.transforms"]

setup(name='nornir_imageregistration',
      version='1.0',
      description="Contains the core image registration algorithms for aligning 2d images into larger mosaics and 3D volumes",
      author="James Anderson",
      author_email="James.R.Anderson@utah.edu",
      url="https://github.com/jamesra/nornir_imageregistration",
      packages=packages,
      requires=required_packages)
