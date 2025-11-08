import numpy as np
import matplotlib.pyplot as plt 
import qutip as qtp
from qutip import basis, tensor
import jax
import jax.numpy as jnp
import jax.numpy.linalg  as nlg
from jax import grad
from jax import jit
from jax.example_libraries import optimizers
from jax import config
config.update("jax_enable_x64", True)
import optax
from tqdm.auto import tqdm
import time


@jit
def softmax(probs: jnp.ndarray):
    # soft max function for the probabilities
    ex1 = jnp.exp(probs)
    sumex = jnp.sum(ex1)
    return jnp.divide(ex1, sumex)

def Nkets(n: int, hilbert: int):
    # random n kets with hilbert space hilbert
    l:list = []
    for i in range(n):
        l.append(qtp.rand_ket(hilbert))
    return l

@jit
def expect_ket(ope, k_p_l: jnp.ndarray, prob: jnp.ndarray):
    # calculates the expected value with the 2 list C_l and P_l
    HS = len(ope[0])
    kn = int(len(k_p_l)/HS)
    def sub_exp(ope):
        sum1_m = 0
        for i in range(kn):
            v = k_p_l[i*HS:(i+1)*HS]
            d1 = jnp.conj(v.T)@ope@v
            d1 = jnp.real(prob[i]*d1)
            sum1_m += d1
        return sum1_m
    expf = jax.vmap(sub_exp)(ope)
    expf = expf.flatten()
    return expf

# @jit
def jnpunit(ket: jnp.ndarray, HS:int, kn:int):
    # to check the unity and for constrain
    list_ke = []
    for i in range(kn):
        ket_i = ket[i*HS:(i+1)*HS]
        norm1 = jnp.linalg.norm(ket_i)
        list_ke.append(ket_i/norm1)
    return jnp.vstack(list_ke)

def rho_stat(ket, probs, HS:int, kn:int):
    # create the density matrix from the two lists
    mix1 = 0
    for i in range(kn):
        v_k = qtp.Qobj(ket[i*HS:(i+1)*HS])
        den_i = v_k*v_k.dag()
        mix1 += probs[i]*den_i
    return mix1

@jit
def cost(rho1: jnp.ndarray, probs: jnp.ndarray,data: jnp.ndarray, ops_jnp: jnp.ndarray, lamb:float):
    """
    Return the cost function to do GD 
    rho1: Is the guess lower-triangular descomposition of the density matrix
    ops_jnp: POVM
    data: data of the measurement of original rho
    """
    l1 = jnp.sum((data - expect_ket(ops_jnp,rho1, probs))**2)
    return l1 + lamb*jnp.linalg.norm(rho1, 1)


def gd_project(data, rho_or, ops_jnp: jnp.ndarray, params: jnp.ndarray, prob:jnp.ndarray, 
        iterations: int, batch_size: int, lr: float = 1e-1, decay:float = 0.999, lamb:float = 0.0000001, tqdm_off=False):
  """
  Function to do the GD-proj.
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
  
  
  # pbar_GD = tqdm(range(iterations)) 
  start_learning_rate = lr
  optimizer = optax.adamax(learning_rate=lr)
  # Exponential decay of the learning rate.
  scheduler = optax.exponential_decay(
      init_value=start_learning_rate, 
      transition_steps=iterations,
      decay_rate=decay)
  # Combining gradient transforms using `optax.chain`.
  gradient_transform = optax.chain(
      optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
      optax.scale_by_adam(),  # Use the updates from adam.
      optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
      # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
      optax.scale(-1.0)
  )
  opt_state_k = gradient_transform.init(params)
  opt_state_p = optimizer.init(prob)
  num_me = len(data)
  # opt_state = optimizer.init(params)
  if not tqdm_off:
    pbar_GD = tqdm(range(iterations)) 
  HS = len(ops_jnp[0])
  kn = int(len(params)/HS)
  loss1 = []
  fidelities_GD = []
  timel_GD = []
  
  @jit
  def step(params, prob, data, ops_jnp, opt_state_k, opt_state_p):
    grad_k = jax.grad(cost, argnums=0)(params, prob, data, ops_jnp, lamb)
    grad_p = jax.grad(cost, argnums=1)(params, prob, data, ops_jnp, lamb)
    grad_k = jnp.conj(grad_k)
    grad_p = jnp.conj(grad_p)
    # ---- kets -------------------------------------------------------------------------
    # grad_f = low_cons(grad_f)   
    updates_k, opt_state_k = gradient_transform.update(grad_k, opt_state_k, params)
    params = optax.apply_updates(params, updates_k)
    # ----- probs ----------------------------------------------------------------------
    updates_p, opt_state_p = optimizer.update(grad_p, opt_state_p, prob)
    prob = optax.apply_updates(prob, updates_p)
    
    return params, prob, opt_state_k, opt_state_p
  
  tot_time = 0
  for i in tqdm(range(iterations), disable=tqdm_off):
    start = time.time()
    
    indix = np.random.randint(0, num_me, size=[batch_size])
    data_b = jnp.asarray(data[[indix]].flatten())
    ops2 = ops_jnp[indix]
    params, prob, opt_state_k, opt_state_p = step(params, prob, data_b, ops2, opt_state_k, opt_state_p)
    # params = rho_cons(params)
    params = jnpunit(params, HS, kn)
    prob = jnp.asarray(softmax(prob))
    loss1.append(float(cost(params, prob, data, ops_jnp, lamb)))
    #...............................................................
    par1 = rho_stat(params, prob, HS, kn)
    # a += verific(par1)
    #.................................................................
    f = qtp.fidelity(rho_or, qtp.Qobj(par1))
    # f = qtp.fidelity(rho_or, qtp.Qobj(params))
    fidelities_GD.append(f)
    end = time.time()
    timestep = end - start
    tot_time += timestep
    timel_GD.append(tot_time)
    #timel_GD.append(end - start) 

    if not tqdm_off: 
        pbar_GD.set_description("Fidelity GD-projection {:.4f}".format(f))
        pbar_GD.update()

  params1 = rho_stat(params, prob, HS, kn)
  return params1, fidelities_GD, timel_GD, loss1