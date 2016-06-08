# -*- coding: utf-8 -*-
"""
Created on 09.05.16
Created for pyExtremeLM

Based on: https://github.com/pypa/sampleproject

@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de

    Copyright (C) {2016}  {Tobias Sebastian Finn}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
from distutils.core import setup
from codecs import open
from os import path

# External modules

# Internal modules

__version__ = "0.2"


setup(
    name='pyextremelm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.2',

    description='An implementation of Extreme Learning Machines for python',

    # The project's main homepage.
    url='https://github.com/maestrotf/pyExtremeLM',

    # Author details
    author='Tobias Sebastian Finn',
    author_email='t.finn@meteowindow.com',

    # Choose your license
    license='GPL3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
#     classifiers=[
#         # How mature is this project? Common values are
#         #   3 - Alpha
#         #   4 - Beta
#         #   5 - Production/Stable
#         'Development Status :: 3 - Alpha',
#
#         # Pick your license as you wish (should match "license" above)
#         'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
#
#         # Specify the Python versions you support here. In particular, ensure
#         # that you indicate whether you support Python 2, Python 3 or both.
#         'Programming Language :: Python :: 3.5',
#
#         'Topic :: Scientific / Engineering',
#         'Topic :: Scientific/Engineering :: Artificial Intelligence',
#         'Topic :: Scientific/Engineering :: Information Analysis',
#         'Topic :: Software Development :: Libraries :: Python Modules',
# ],

    # What does your project relate to?
    keywords='machine learning neural network ann elm',
    packages=["pyextremelm"],
    package_dir={"pyextremelm": "pyextremelm"},

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    install_requires=['numpy', 'sklearn', 'scipy']
)
