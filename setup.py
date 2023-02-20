# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:47:50 2022

@author: m.debritovanvelze
"""

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'A python package to process calcium imaging data'
LONG_DESCRIPTION = 'This package takes the output from Suite2 and Facemap and combines it with locomotion, whisking and other data.'

# Setting up
setup(
        name="CalciumImagingAnalysisPackage", 
        version=VERSION,
        author="Marcel van Velze",
        author_email="<marcelbvv@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'Calcium Analysis', 'Bruker', 'Suite2P', 'Facemap'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)