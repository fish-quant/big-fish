# -*- coding: utf-8 -*-

"""
Setup script.
"""

from setuptools import setup, find_packages

# TODO remove useless packages (numba and umap)

# Package meta-data.
VERSION = 1.1
DESCRIPTION = 'Toolbox for cell FISH images.'

# Package abstract dependencies
REQUIRES = [
      'numpy >= 1.16.0',
      'pip >= 18.1',
      'scikit-learn >= 0.20.2',
      'scikit-image >= 0.14.2',
      'scipy >= 1.2.0',
      'matplotlib >= 3.0.2',
      'pandas >= 0.24.0',
      'numba >= 0.37.0',
      'umap-learn >= 0.3.9',
      'mrc >= 0.1.5'
]

DEEPLEARNING_REQUIREMENTS = ['tensorflow >= 1.12.0, < 2.0']

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
      url='https://github.com/fish-quant/big-fish',
      packages=find_packages(include=['bigfish']),
      license='MIT',
      python_requires='>=3.6.0',
      install_requires=REQUIRES,
      extras_require={
            'deeplearning': DEEPLEARNING_REQUIREMENTS
      },
      classifiers=CLASSIFIERS
      )
 
