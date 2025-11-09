
from functools import partial
import numpy as np
import matplotlib.pyplot as plt 
import qutip as qtp
from qutip import basis, tensor
from numpy.random import default_rng

import jax
import jax.numpy as jnp
import jax.numpy.linalg  as nlg
from jax import grad
from jax import jit
from jax.example_libraries import optimizers
from jax import config
config.update("jax_enable_x64", True)


from tqdm.auto import tqdm
import time



@jit
def stiefel_update(params,
                   grads,
                   step_size):
    """
    This is for retraction procedure to maintain TP constraint 
    """
    U = jnp.hstack([grads, params])
    V = jnp.hstack([params, -grads])

    prod_term = V.T.conj()@U
    invterm = jnp.eye(prod_term.shape[0]) + (step_size/2.)*prod_term
    A = step_size*(U@jnp.linalg.inv(invterm))

    B = V.T.conj()@params

    updated_params = params - A@B
    return updated_params

def softmax(probs):
    #Softmax function for the probabilities
    ex1 = np.exp(probs)
    sumex = np.sum(ex1)
    return np.divide(ex1, sumex)

def Nkets(n: int, hilbert: int):
    # create n random kets, with hilbert space hilbert
    l:list = []
    for i in range(n):
        l.append(qtp.rand_ket(hilbert))
    return l

@jit
def expect_prob_ket(ope: jnp.ndarray, k_p_l: jnp.ndarray):
    # Calculates the expected value with the representation of the ansatz W
    HS = len(ope[0])
    kn = int(len(k_p_l)/HS)
    def sub_exp(ope):
        sum1_m = 0
        for i in range(kn):
            v = k_p_l[i*HS:(i+1)*HS]
            d1 = jnp.conj(v.T)@ope@v
            d1 = jnp.real(d1)
            sum1_m += d1
        return sum1_m
    expf = jax.vmap(sub_exp)(ope)
    expf = expf.flatten()
    return expf

def mix_rho(ket_M_l: jnp.ndarray, HS: int, kn:int):
    # to go from the random kets to the density matrix
    mixed_density_2 = 0
    for i in range(kn):
        v_k = ket_M_l[i*HS:(i+1)*HS]
        v_k = qtp.Qobj(v_k)
        density_i = v_k*v_k.dag()
        mixed_density_2 += density_i
    return mixed_density_2

#cost function
@jit 
def cost(rho1: jnp.ndarray, data: jnp.ndarray, ops_jnp: jnp.ndarray, lamb:float):
    """
    Return the cost function to do GD 
    rho1: Is the guess lower-triangular descomposition of the density matrix
    ops_jnp: POVM
    data: data of the measurement of original rho
    """
    l1 = jnp.sum((data - expect_prob_ket(ops_jnp,rho1))**2)
    return l1 + lamb*jnp.linalg.norm(rho1, 1)

def gd_manifold(data, ops_jnp, params: jnp.ndarray, iterations: int, 
                batch_size: int, rho_or = None, lr: float =0.1,decay: float = 0.99999, lamb:float =0.0001, tqdm_off=False):
  """
  Function to do the GD-mani.
  Return:
    params1: The reconstructed density matrix
    fidelities_GD: A list with the fidelities values per iteration
    timel_GD: A list with the value of the time per iteration
    loss1: A list with the value of the loss function per iteration

  Input:
    data: the expected value of the original density matrix
    rho_or: original density matrix, to calculate the fidelity
    ops_jnp: POVM in jnp array
    params: Ansatz, the W vector 
    iterations: number of iterations for the method
    batch_size: batch size
    lr: learning rate
    decay: value of the decay of the lr
    lamb: hyperparameter for the penalization
    tqdm_off: To show the iteration bar. True is to desactivate (for the cluster)
    
  """
  
  if not tqdm_off:
    pbar_GD = tqdm(range(iterations)) 
  
  HS = len(ops_jnp[0])
  kn = int(len(params)/HS)
  alpha = decay
  initial1 = lr
  loss1 = []
  fidelities_GD = []
  timel_GD = []
  #rho_in = mix_rho(params, HS, kn)
  #fidelities_GD.append(qtp.fidelity(rho_or, rho_in))
  num_me = len(data)
  #loss1.append(float(cost(params, jnp.asarray(data), ops_jnp, lamb)))
  
  @jit
  def step(params, initiald, data, ops_jnp):
    grad_f = jax.grad(cost, argnums=0)(params,data, ops_jnp, lamb)
    grads = jnp.conj(grad_f)           # do a conjugate, if not can create some problems
    grads = grads/jnp.linalg.norm(grads)
    params = stiefel_update(params, grads, initiald)
    # print(initiald)
    # initiald = alpha*initiald
    return params

  tot_time = 0
  for i in tqdm(range(iterations), disable=tqdm_off):

    start = time.time()

    rng = default_rng()
    indix = rng.choice(num_me, size=batch_size, replace=False)
    # indix = np.random.randint(0, num_me, size=[batch_size])
    data_b = jnp.asarray(data[[indix]].flatten())
    ops2 = ops_jnp[indix]
    params = step(params, initial1, data_b, ops2)
    initial1 = initial1*alpha
    # print(initial1)
    loss1.append(float(cost(params, data_b, ops2, lamb)))
    rho_iter = mix_rho(params, HS, kn)
    
    if rho_or is not None:
        f = qtp.fidelity(rho_or, qtp.Qobj(rho_iter))
        fidelities_GD.append(f)

    end = time.time()
    timestep = end - start
    tot_time += timestep
    timel_GD.append(tot_time)
    #timel_GD.append(end - start) 

    if not tqdm_off:
        pbar_GD.set_description("Fidelity GD-manifold {:.4f}".format(f))
        pbar_GD.update()


  params1 = mix_rho(params, HS,kn)
  return params1, fidelities_GD, timel_GD, loss1