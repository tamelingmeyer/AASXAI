# Analysing and Comparing Automated Algorithm Selection Models using Explainable Artificial Intelligence

Due to the size limit this repository contains only the data for the scenario BNSL-2016.

## Setup Environment
The folder conda contains the conda environment used (LINUX). You can recreate it with:

`conda env create --file environment.yml`

Additionaly the two git repositories in additional_pip_packages need to be installed via pip install.

`cat additional_pip_packages.txt | xargs -n 1 -L 1 pip install`

## Visulisations
All plots visualizing the Shapley values are generated in Visualisations.ipynb. (paper Section 3)

## Comparisons
All plots used to compare different models are generated in Comparisons.ipynb. (paper Section 4)


## Models
The folder models contains all models that are used in the paper. They were generate by the script slurm_compute_af_models.sh.

## Shapley values
shap_values contains all used Shapley values as .npy. They were computed using slurm_compute_shaps_af.sh. 
