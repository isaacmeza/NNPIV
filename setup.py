# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnpiv",  
    version="0.2.0",
    author="",
    author_email="",
    description="NESTED NONPARAMETRIC INSTRUMENTAL VARIABLE REGRESSION",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['cvxopt>1.2.0']
)
