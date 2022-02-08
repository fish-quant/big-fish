# Big-FISH

[![PyPI version](https://badge.fury.io/py/big-fish.svg)](https://badge.fury.io/py/big-fish)
[![Build Status](https://travis-ci.com/fish-quant/big-fish.svg?branch=master)](https://travis-ci.com/fish-quant/big-fish)
[![Documentation Status](https://readthedocs.org/projects/big-fish/badge/?version=stable)](https://big-fish.readthedocs.io/en/latest/?badge=stable)
[![codecov](https://codecov.io/gh/fish-quant/big-fish/branch/master/graph/badge.svg)](https://codecov.io/gh/fish-quant/big-fish)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green)](https://github.com/fish-quant/big-fish/blob/master/LICENSE)
[![Python version](https://img.shields.io/pypi/pyversions/big-fish.svg)](https://pypi.python.org/pypi/big-fish/)

**Big-FISH** is a python package for the analysis of smFISH images. It includes various methods to **analyze microscopy images**, such **spot detection** and **segmentation of cells and nuclei**. The package allows the user represent the extract properties of a cell as coordinates (see figure below). The ultimate goal is to simplify **large scale statistical analysis** and quantification.

| Cell image (smFISH channel) and its coordinates representation |
| ------------- |
| ![](images/plot_cell.png "Nucleus in blue, mRNAs in red, foci in orange and transcription sites in green") |

## Installation

### Dependencies

Big-FISH requires Python 3.6 or newer. Additionally, it has the following dependencies:

- numpy (>= 1.16.0)
- scipy (>= 1.4.1)
- scikit-learn (>= 0.24.0)
- scikit-image (>= 0.14.2)
- matplotlib (>= 3.0.2)
- pandas (>= 0.24.0)
- mrc (>= 0.1.5)

For segmentation purpose, two additional dependencies can be requested:
- tensorflow (== 2.3.0)
- tensorflow-addons (== 0.12.1)

### Virtual environment

To avoid dependency conflicts, we recommend the the use of a dedicated [virtual](https://docs.python.org/3.6/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment.  In a terminal run the command:

```bash
conda create -n bigfish_env python=3.6
source activate bigfish_env
```

We recommend two options to then install Big-FISH in your virtual environment.

#### Download the package from PyPi

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Big-FISH. In a terminal run the command:

```bash
pip install big-fish
```

#### Clone package from Github

Clone the project's [Github repository](https://github.com/fish-quant/big-fish) and install it manually with the following commands:

```bash
git clone git@github.com:fish-quant/big-fish.git
cd big-fish
pip install .
```

## Usage

Big-FISH provides a toolbox for the full analysis pipeline of smFISH images. A complete [documentation](https://big-fish.readthedocs.io/en/stable/) is available online. 

This package is part of the [FISH-Quant](https://fish-quant.github.io/) framework and several examples are also available as [Jupyter notebooks](https://github.com/fish-quant/big-fish-examples/tree/master/notebooks).

## Support

If you have any question relative to the repository, please open an [issue](https://github.com/fish-quant/big-fish/issues). You can also contact [Arthur Imbert](mailto:arthur.imbert@mines-paristech.fr) or [Florian Mueller](mailto:muellerf.research@gmail.com).

## Roadmap (suggestion)

Version 1.0.0:
- Complete code coverage.
- Unpin deep learning dependencies
- Add a pretrained pattern recognition model

## Development

### Source code

You can access the latest sources with the commands:

```bash
git clone git@github.com:fish-quant/big-fish.git
cd big-fish
git checkout develop
```

### Contributing

[Pull requests](https://github.com/fish-quant/big-fish/pulls) are welcome. For major changes, please open an [issue](https://github.com/fish-quant/big-fish/issues) first to discuss what you would like to change.

### Testing

Please make sure to update tests as appropriate if you open a pull request. You can install exacts dependencies and specific version of [pytest](https://docs.pytest.org/en/latest/) by running the following command:

```bash
pip install -r requirements_dev.txt
```

To perform unit tests, run : 

```bash
pytest bigfish
```

## Citation

If you exploit this package for your work, please cite:

> Arthur Imbert, Wei Ouyang, Adham Safieddine, Emeline Coleno, Christophe Zimmer, Edouard Bertrand, Thomas Walter, Florian Mueller. FISH-quant v2: a scalable and modular analysis tool for smFISH image analysis. bioRxiv (2021) https://doi.org/10.1101/2021.07.20.453024
