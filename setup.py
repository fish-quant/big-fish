# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Setup script.
"""

from setuptools import setup, find_packages

# package description
DESCRIPTION = "Toolbox for the analysis of smFISH images."


# package version
VERSION = None
with open('bigfish/__init__.py') as f:
    for row in f:
        if row.startswith('__version__'):
            VERSION = row.strip().split()[-1][1:-1]
            break

# package abstract dependencies
REQUIREMENTS = [
    'numpy >= 1.16.0',
    'pip >= 18.1',
    'scikit-learn >= 0.20.2',
    'scikit-image >= 0.14.2',
    'scipy >= 1.2.0',
    'matplotlib >= 3.0.2',
    'pandas >= 0.24.0',
    'mrc >= 0.1.5']
DEEPLEARNING_REQUIREMENTS = [
    'tensorflow >= 1.12.0, < 2.0']

# long description of the package
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# A list of classifiers to categorize the project (only used for searching and
# browsing projects on PyPI)
CLASSIFIERS = [
    'Development Status :: 2 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Intended Audience :: Biologist',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Topic :: Cellular Imagery',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: MIT License']

# setup
setup(name='big-fish',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author='Arthur Imbert',
      author_email='arthur.imbert.pro@gmail.com',
      url='https://github.com/fish-quant/big-fish',
      packages=find_packages(),
      license='BSD 3-Clause License',
      python_requires='>=3.6.0',
      install_requires=REQUIREMENTS,
      extras_require={'deeplearning': DEEPLEARNING_REQUIREMENTS},
      classifiers=CLASSIFIERS)
