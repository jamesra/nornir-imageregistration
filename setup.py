'''
Created on Aug 30, 2013

@author: u0490822
'''




from ez_setup import use_setuptools
from setuptools import setup, find_packages
import os
import glob

if __name__ == '__main__':
    use_setuptools()

    packages = ["nornir_imageregistration",
                "nornir_imageregistration.io",
                "nornir_imageregistration.geometry",
                "nornir_imageregistration.transforms"]

    install_requires = ["nornir_pools",
                        "nornir_shared",
                        "numpy>=1.7.1",
                        "scipy>=0.12",
                        "matplotlib"]

    dependency_links = ["git+http://github.com/jamesra/nornir-pools#egg=nornir_pools",
                        "git+http://github.com/jamesra/nornir-shared#egg=nornir_shared"]

    scripts = glob.glob(os.path.join('scripts', '*.py'))

    setup(name='nornir_imageregistration',
          version='1.0',
          description="Contains the core image registration algorithms for aligning 2d images into larger mosaics and 3D volumes",
          author="James Anderson",
          author_email="James.R.Anderson@utah.edu",
          url="https://github.com/jamesra/nornir-imageregistration",
          packages=packages,
          scripts=scripts,
          test_suite='test',
          install_requires=install_requires,
          test_requires=install_requires,
          dependency_links=dependency_links )
