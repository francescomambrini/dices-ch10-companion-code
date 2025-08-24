# Companion Code repository for *The (annotated) language of the Homeric Heroes*

This repository contains the code and data used to support my Chapter 10 in the edited volume of the [DICES](https://www.dices.uni-rostock.de/en/about-dices/) network, entitled *Direct speech in Greek and Latin epic: expanding the methods and Canon*.

The title of my chapter is: "The (annotated) language of the Homeric Heroes. Towards a treebank-based approach".

## Structure

The main code for the analysis (with some explanation) is in the `analysis.ipynb`. The file is a Jupyter Notebook which was originally written with Python 3.10, and was tested in a dedicated virtual environment using 3.13.2. The libraries needed to run the code are listed in the `requirements.txt` file (the version numbers refer to the environment with Python 3.13.2). Any environment with Python >= 3.10 (and probably even earlier versions of Python 3) should be fine

Other directories:
- `data/`: contains the dataframes and treebank files that are needed for the analyses;
- `character_utils/`: a python module that contains some of the functions used in the code; most of them (but not all) are also re-defined in the `analysis.ipynb` notebook for readability;
- `figures/`: contains some of the plots generated for the publication and some other image used in the notebook.

## License

Data (including the figures generated with the code) and code are released under a [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)license (see the LICENSE file for the full text).
