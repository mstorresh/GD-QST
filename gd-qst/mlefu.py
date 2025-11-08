import numpy as np
import qutip as qtp

from tqdm.auto import tqdm
import time



def mle_CV(data, rho_or, Measu_ope, rho, max_itera: int, tqdm_off=False):
    """
    Quantum State Tomography with MLE for Continous Variable.
    Original code from https://github.com/quantshah/qst-cgan , whit some adaptations.
    
    Args: 
        data (list): List of original measurement data
        rho_or (qtp Qobj): Original density matrix
        Measu_ope (list): list with the measurement operators (povm) in qtp.Qobj
        rho (qtp Qobj): ansatz density matrix (random density matrix)
        max_itera (int): number of iterations
        
    Returns:
        rho (qtp Qobj): Final rho after doing mle qst 
        fidelities_MLE (list): list with the fidelities trhough the iterations 
        timel_MLE (list): time need it per loop  
    """
    ops_np = [op.full() for op in Measu_ope]
    fidelities_MLE = []
    timel_MLE = []
    fidelities_MLE.append(qtp.fidelity(rho_or, rho))
    if not tqdm_off:
        pbar_MLE = tqdm(range(max_itera))
    
    tot_time = 0

    for i in range(max_itera):
        start = time.time()
        
        guessed_val = qtp.expect(Measu_ope, rho)
        ratio = data / guessed_val
        R = qtp.Qobj(np.einsum("aij,a->ij", ops_np, ratio))
        rho = R * rho * R
        rho = rho / rho.tr()
        f = qtp.fidelity(rho_or, rho)
        fidelities_MLE.append(f)
        
        end = time.time()
        timestep = end - start
        tot_time += timestep
        timel_MLE.append(tot_time)
        if not tqdm_off:
            pbar_MLE.set_description("Fidelity iMLE {:.4f}".format(f))
            pbar_MLE.update()
    
    return rho, fidelities_MLE, timel_MLE


def mle_dv(data, rho_or, Measu_ope, rho, max_itera: int, tqdm_off=False):
    
    """
    Quantum State Tomography with MLE for Continous Variable.
    Original code from https://github.com/quantshah/qst-cgan , whit some adaptations.
    
    Args: 
        data (list): List of original measurement data
        rho_or (qtp Qobj): Original density matrix
        Measu_ope (list): list with the measurement operators (povm) in qtp.Qobj
        rho (qtp Qobj): ansatz density matrix (random density matrix)
        max_itera (int): number of iterations
        
    Returns:
        rho (qtp Qobj): Final rho after doing mle qst 
        fidelities_MLE (list): list with the fidelities trhough the iterations 
        timel_MLE (list): time need it per loop  
    """
    
    count1 = 0
    ops_np = [op.full() for op in Measu_ope]
    fidelities_MLE = []
    timel_MLE = []
    fidelities_MLE.append(qtp.fidelity(rho_or, rho))
    if not tqdm_off:
        pbar_MLE = tqdm(range(max_itera))
    
    for i in range(max_itera):
        start = time.time()
        guessed_val = qtp.expect(Measu_ope, rho)
        ratio = data / guessed_val
        for a in range(len(guessed_val)):
            val1 = float(guessed_val[a])
            val2 = float(data[a])
            # print(a)
            if (val1==0.0 or np.abs(val1)<1e-10)  and val2==0.0:
                ratio[a] = 1.0
            else:    
                if val1==0.0 or np.abs(val1)<1e-15:
                    count1 +=1
                else: 
                    pass
        if count1>0:
            print("There is one or more probability 0, and in these cases is when the fidelity is close to 1. So, we brake the loop")
            break
        R = qtp.Qobj(np.einsum("aij,a->ij", ops_np, ratio))
        rho = R * rho * R
        rho = rho / rho.tr()
        f = qtp.fidelity(rho_or, rho)
        fidelities_MLE.append(f)
        end = time.time()
        timel_MLE.append(end - start)
        if not tqdm_off:
            pbar_MLE.set_description("Fidelity iMLE {:.4f}".format(f))
            pbar_MLE.update()
    
    return rho, fidelities_MLE, timel_MLE
