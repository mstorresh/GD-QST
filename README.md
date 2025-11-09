# Quantum state tomography with gradient descent 

This repository contains the code required to reproduce the plots from https://arxiv.org/abs/2503.04526  

## Installation and use

Install the environment.yml or you can install the libraries by yourself, but be careful with the version of QuTip, in our case we are using qutip==5.0.0a2

```python
conda env create -f environment.yml
conda activate gd_qst_env
```
The notebooks in the `examples` folder are tutorials on how to do the quantum state tomography with the different methods of gradient descent. 

The `data_and_paper-plots` folder contains a Jupyter Notebook (`.ipynb`) that runs the methods used to generate the plots presented in the paper, along with a ZIP file containing the corresponding data.

## 🔹 Installation

Install the package directly from GitHub:

```bash 

pip install git+https://github.com/mstorresh/GD-QST.git@pip-gd-qst

## ⚠️ Tested environment:
- Python 3.10
- QuTiP 5.0.0a2
- NumPy 1.26.3
- SciPy 1.11.4