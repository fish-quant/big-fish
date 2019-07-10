# -*- coding: utf-8 -*-

"""
Setup script.
"""

from setuptools import setup, find_packages

# Package meta-data.
VERSION = 1.0
DESCRIPTION = 'Toolbox for cell FISH images.'

# Package abstract dependencies
REQUIRES = [
      'numpy',
      'scikit-learn',
      'scikit-image',
      'scipy',
      'pandas',
      'tensorflow',
      'matplotlib',
      'joblib'
]

# Long description of the package
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# A list of classifiers to categorize the project (only used for searching and
# browsing projects on PyPI).
CLASSIFIERS = [
      'Development Status :: 3 - Alpha',
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
      'License :: OSI Approved :: MIT License'
]

# Setup
setup(name='big-fish',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author='Arthur Imbert',
      author_email='arthur.imbert.pro@gmail.com',
      url='https://github.com/Henley13/big-fish',
      packages=find_packages(),
      license='MIT',
      python_requires='>=3.6.0',
      install_requires=REQUIRES,
      classifiers=CLASSIFIERS
      )
