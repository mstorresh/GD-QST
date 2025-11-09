from qutip import * 
import qutip as qt

import qutip.qip.operations.gates as qugate # type: ignore

import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import *
import random
import cmath as cm
from scipy.optimize import minimize
import cvxpy as cp
import random as rand
import time
############## 

def cvx_qst(measurement_ops, basis_set, B, gamma, dimension, rho_ideal):

    start = time.time()
    # Initialize the sensing matrix A
    A = np.empty((len(measurement_ops), len(basis_set)), dtype=complex)

    # Compute the sensing matrix A using the merged functionality
    for i in range(len(measurement_ops)):
        for j in range(len(basis_set)):
            A[i, j] = (measurement_ops[i] * basis_set[j]).tr()

    # Define the variable for the quantum state (rho)
    X = cp.Variable((dimension, dimension), hermitian=True)

    # Define the objective function to minimize
    obj = cp.Minimize(cp.norm(A @ cp.reshape(X.T, (dimension**2,)) - B) + gamma*cp.norm(cp.reshape(X.T, (dimension**2,)), 1))

    # Define the constraints: positive semidefinite and trace equal to 1
    constraints = [cp.trace(X) == 1, X >> 0]

    # Set up and solve the optimization problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Extract the result as a quantum object
    rho_cvx = Qobj(X.value, dims=([[dimension], [dimension]]))

    # Fidelity between rho_LS and rho_or
    f_cvx = fidelity(rho_cvx, rho_ideal)

    end = time.time()
    total_time = end - start

    return rho_cvx, f_cvx, total_time


    
    


