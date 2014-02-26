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
                "nornir_imageregistration.files",
                "nornir_imageregistration.geometry",
                "nornir_imageregistration.transforms"]

    install_requires = ["nornir_pools>=1.1.1",
                        "nornir_shared>=1.1.2",
                        "numpy>=1.8",
                        "scipy>=0.13.2",
                        "matplotlib",
                        "pillow>=2.3",
                        "rtree>=0.7"]

    dependency_links = ["git+http://github.com/nornir/nornir-pools#egg=nornir_pools-1.1.1",
                        "git+http://github.com/nornir/nornir-shared#egg=nornir_shared-1.1.2"]

    scripts = glob.glob(os.path.join('scripts', '*.py'))

    setup(name='nornir_imageregistration',
          version='1.1.2',
          description="Contains the core image registration algorithms for aligning 2d images into larger mosaics and 3D volumes",
          author="James Anderson",
          author_email="James.R.Anderson@utah.edu",
          url="https://github.com/nornir/nornir-imageregistration",
          packages=packages,
          scripts=scripts,
          test_suite='test',
          install_requires=install_requires,
          dependency_links=dependency_links)