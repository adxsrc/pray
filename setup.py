#!/usr/bin/env python3
#
# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of PRay.
#
# PRay is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PRay is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PRay.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, find_packages


setup(
    name='pray',
    version='0.1.1',
    url='https://github.com/adxsrc/pray',
    author='Chi-kwan Chan',
    author_email='chanc@arizona.edu',
    description='General relativistic ray tracing in python based on JAX',
    packages=find_packages('mod'),
    package_dir={'': 'mod'},
    python_requires='>=3.7',
    install_requires=[
        'fadge==0.1.1',
    ],
)
