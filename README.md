#  Quantum State Tomography with Gradient Descent

This repository contains the code required to reproduce the results and plots from the following paper:

📄 [IOP Quantum Science and Technology (2025)](https://iopscience.iop.org/article/10.1088/2058-9565/ae0baa)  
📰 [arXiv:2503.04526](https://arxiv.org/abs/2503.04526)

---

## 🚀 Overview

This project implements **Quantum State Tomography (QST)** using different **gradient descent methods**.  
It includes reproducible examples, the full dataset used in the publication, and scripts to regenerate the figures presented in the paper.

---

## ⚙️ Installation & Usage

### Option 1 — Cloning the full repository 

This option ensures full reproducibility of the results and provides access to all source code and functions.

1. **Clone the repository** to obtain the Python modules and notebooks:

   ```bash
   git clone https://github.com/mstorresh/GD-QST.git
   cd GD-QST
   ```

2. Create and activate the environment from the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    conda activate gd_qst_env
    ```

⚠️ Important: The code is designed to work with a specific QuTiP version.
Please ensure you are using qutip==5.0.0a2 to avoid compatibility issues.

The notebooks in the `examples` folder are tutorials on how to do the quantum state tomography with the different methods of gradient descent. 

The `data_and_paper-plots` folder contains a Jupyter Notebook (`.ipynb`) that runs the methods used to generate the plots presented in the paper, along with a ZIP file containing the corresponding data.

### Option 2 — Installation via pip

If you only need to use the library (without cloning the full repository), you can install it directly from GitHub.

It is recommended to install it in a new virtual environment: 

```bash 

pip install git+https://github.com/mstorresh/GD-QST.git@pip-gd-qst

```
*Creating a new virtual environment with the `environment.yml` file is recommended for a clean setup.*

The pip-installable version is maintained in the branch pip-gd-qst, currently at version 1.0.0.

## 📘 Repository Structure

### Main Branch

- **`data_and_paper-plots/`** — Contains:
  - The Jupyter Notebook used to reproduce the plots from the paper.
  - A ZIP file with the corresponding data.  

- **`examples/`** — Jupyter notebooks demonstrating how to perform QST using various gradient descent methods.  

- **`qst_tec/`**  — Core Python modules containing the implementation of the gradient descent algorithms and QST routines.

- **`test/`**  — Jupyter notebooks to test the gd_qst library installed with pip.

### pip-gd-qst Branch

- **`gd_qst/`**  — Core Python modules containing the implementation of the gradient descent algorithms and QST routines.

- **`test/`**  — Jupyter notebooks to test the gd_qst library installed with pip.


## ⚠️ Tested environment:
- Python 3.10
- QuTiP 5.0.0a2
- NumPy 1.26.3
- SciPy 1.11.4


## 💬 Contact

For questions or comments, feel free to contact me directly or use the corresponding author’s email provided in the paper.

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{gaikwad2025gradient,
  title     = {Gradient-descent methods for fast quantum state tomography},
  author    = {Gaikwad, Akshay and Torres, Manuel Sebastian and Ahmed, Shahnawaz and Kockum, Anton Frisk},
  journal   = {Quantum Science and Technology},
  volume    = {10},
  number    = {4},
  pages     = {045055},
  year      = {2025},
  publisher = {IOP Publishing},
  doi       = {10.1088/2058-9565/ae0baa},
  note      = {Published 8 October 2025},
}
```

Gaikwad, A., Torres, M. S., Ahmed, S., & Kockum, A. F. (2025). Gradient-descent methods for fast quantum state tomography. Quantum Science and Technology, 10(4), 045055.
