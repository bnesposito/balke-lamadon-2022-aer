import argparse, logging
from pathlib import Path
import json
import time
import IPython

import jax
import jax.numpy as jnp
import numpy as np

import opt_einsum as oe

import wagedyn as wd
from wagedyn.search import JobSearchArray
from wagedyn.valuefunction import PowerFunctionGrid
from wagedyn.probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix

import replication_utils as ru

import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info('Start replication')

# ----- parsing input arguments -------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-p","--parameter",  help="parameter file to use (json)")
parser.add_argument("-d","--default",  help="default parameter file to use (json)")
parser.add_argument("-m","--model",  help="file where to save the model",default="")
parser.add_argument("-v","--vdec",  help="file where to save the variance decomposition",default="")
parser.add_argument("-s","--simulate",  help="file where to save simulation results",default="")
parser.add_argument("-l","--logfile",  help="file to log to")
parser.add_argument("-pl","--plot",  help="plot?")
args = parser.parse_args()

param_file = Path(args.parameter)
logger.info("loading file {}".format(param_file))

param_default_file = Path(args.default)
logger.info("loading file {}".format(param_default_file))

plot_flag = bool(int(args.plot))
logger.info("plot flag: {}".format(plot_flag))

# Load specific parameters
pdict_overwrite = json.loads(param_file.read_text())
pdict_default = json.loads(param_default_file.read_text())
pdict = {**pdict_default, **pdict_overwrite}
#pdict = pdict_overwrite
p = wd.RegisteredParameters(pdict)
#p = wd.Parameters(pdict)
print(jax.tree_util.tree_structure(p))

# set the seed if provided
# key = jax.random.PRNGKey(seed)
if 'rseed' in pdict.keys():
    np.random.seed(pdict['rseed'])

deriv_eps = 1e-3 # step size for derivative
num_x1 = int(p.num_x / p.num_np)

# Worker and Match Productivity Heterogeneity in the Model
X_grid = ru.construct_x_grid(p)   # Create worker productivity grid
Z_grid = ru.construct_z_grid(p)   # Create match productivity grid

logger.info(f"Shape of grid X: {X_grid.shape}")
logger.info(f"Shape of grid Z: {Z_grid.shape}")

# Production Function in the Model
fun_prod = p.prod_a * np.power(X_grid * Z_grid, p.prod_rho)

# Unemployment Benefits across Worker Productivities
unemp_bf = np.ones(p.num_x) * p.u_bf_m

# Transition matrices
X_trans_mat = createBlockPoissonTransitionMatrix(p.num_x/p.num_np, p.num_np, p.x_corr)
Z_trans_mat = createPoissonTransitionMatrix(p.num_z, p.z_corr)

logger.info(f"Shape of X transition matrix: {X_trans_mat.shape}")
logger.info(f"Shape of Z transition matrix: {Z_trans_mat.shape}")

# Value Function Setup
V_grid   = ru.construct_v_array(p, unemp_bf, fun_prod)
sup_wage = ru.inv_utility(p, (1 - p.beta) * V_grid)
J_grid   = -10 * np.ones((p.num_z, p.num_v, p.num_x))
X1_w     = np.kron(np.linspace(0, 1, num_x1), np.ones(p.num_np))

logger.info(f"Shape of V grid: {V_grid.shape}")
logger.info(f"Shape of J grid: {J_grid.shape}")

# Unemployment value function (initial condition)
value_unemp = ru.utility(p, unemp_bf) / p.int_rate

# Probability of finding a job of quality v for a worker of type x (initial condition)
prob_find_vx = np.zeros((p.num_v, p.num_x))

# Probability of quitting for worker type x seeing a value v
prob_quit_vx = np.zeros((p.num_v, p.num_x))

# Equilibrium wage function
ax = np.newaxis
wage_eqm = np.tile(ru.inv_utility(p, V_grid * p.int_rate)[ax, :, ax], (p.num_z, 1, p.num_x))

Vf_U = np.zeros(p.num_x)
w_grid = np.linspace(unemp_bf.min(), fun_prod.max(), p.num_v)


# Solve the model ----------------------------------------------
# Setting up the initial values for the VFI
Ji =  J_grid
ite_prob_vx  =  prob_find_vx

W1i = np.zeros(Ji.shape)
Ui  = np.zeros(p.num_x )

# prepare expectation call
Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, Z_trans_mat.shape, X_trans_mat.shape)
Ex  = oe.contract_expression('b,bx->x', Ui.shape, X_trans_mat.shape)

"""
    Solves for a fixed point in U(x) and J(z, v, x) given the set of problems for the worker and firm in the presence of the free-entry condition. 
    
    This solves a version of the model without on-the-job search and with fixed wages.
    We will use this as a starting value for the main model. First, solve the case with a choice of the quit probability.
    
    :return: Updates the unemployment distribution and probability of finding a job of quality v for a worker of type x
"""
start_time = time.time()

for ite_num in range(2*p.max_iter):
    Ji2 = Ji
    Ui2 = Ui
    W1i2 = W1i

    # we compute the expected value next period by applying the transition rules
    EW1i = Exz(W1i, Z_trans_mat, X_trans_mat)
    EJ1i = Exz(Ji, Z_trans_mat, X_trans_mat)
    EUi  = Ex(Ui, X_trans_mat)

    # we compute quit decision of the worker
    qi = ru.inv_effort_cost_1d(p, - p.beta * (EW1i - EUi))

    # update Ji
    Ji  = fun_prod[:,ax,:] - w_grid[ax,:,ax] + p.beta * (1 - qi) * EJ1i
    Ji = ru.impose_decreasing(Ji)

    # Update worker value function
    W1i = ru.utility(p, w_grid)[ax,:,ax] - ru.effort_cost(p, qi) + \
               p.beta * qi * EUi + p.beta * (1-qi) * EW1i
    # W1i = jax.vmap(ru.utility, (None, 0))(p, w_grid)[ax,:,ax] - jax.vmap(ru.effort_cost, (None,0)) (p, qi) + \
    #             p.beta * qi * EUi + p.beta * (1-qi) * EW1i

    # Apply the matching function
    ite_prob_vx = p.alpha * jnp.power(1 - jnp.power(
        jnp.divide(p.kappa, jnp.maximum(Ji[p.z_0 - 1, :, :], 1.0)), p.sigma), 1/p.sigma)

    # Update the guess for U(x) given p
    Ui = jnp.max( ru.utility_gross(p, unemp_bf[ax, :]) + p.beta * ite_prob_vx *
                        (W1i[p.z_0 - 1, :, :] - EUi[ax, :]) + p.beta * EUi[ax, :], axis=0)

    # Compute the norm-inf between the two iterations of U(x)
    error_u  = jnp.max(abs(Ui - Ui2))
    error_j  = jnp.max(abs(Ji - Ji2))
    error_w1 = jnp.max(abs(W1i - W1i2))

    if np.array([error_u, error_w1, error_j]).max() < p.tol_simple_model and ite_num>10:
        break

    if (ite_num % 25 ==0):
        logger.debug('[{}] Error_U = {:2.4e}, Error_J = {:2.4e}, Error_W1 = {:2.4e}'.format(ite_num, error_u, error_j,error_w1))
    
end_time = time.time()
execution_time = end_time - start_time
logger.info(f"Execution time: {execution_time} seconds")

logger.debug(f"Shape of qi: {qi.shape}")
logger.debug(f"Shape of Ji: {Ji.shape}")
logger.debug(f"Shape of W1i: {W1i.shape}")
logger.debug(f"Shape of ite_prob_vx: {ite_prob_vx.shape}")
logger.debug(f"Shape of Ui: {Ui.shape}")


logger.info('[{}] Error_U = {:2.4e}, Error_J = {:2.4e}, Error_W1 = {:2.4e}'.format(ite_num, error_u, error_j, error_w1))

# extract U2E probability
usearch = np.argmax( ru.utility(p, unemp_bf[ax, :]) + p.beta * ite_prob_vx *
                (W1i[p.z_0 - 1, :, :] - EUi[ax, :]) + p.beta * EUi[ax, :], axis=0)
Pr_u2e = [ ite_prob_vx[usearch[ix],ix] for ix in range(p.num_x) ]

Vf_J    = Ji                        # Job value function
Vf_W1   = W1i                       # Worker value function
Fl_ce   = ru.effort_cost(p, qi)     # Cost of effort
Pr_e2u  = qi                        # Probability of quitting   
Fl_wage = w_grid                    # Wage grid
Vf_U    = Ui                        # Unemployment value function
Pr_u2e  = Pr_u2e                    # Probability of going from unemployment to employment
prob_find_vx = ite_prob_vx          # Probability of finding a job of quality v for a worker of type x

print(Ui)

nrows = 3
ncols = 3

if plot_flag:
    plt.figure(figsize=(16, 12))

    plt.subplot(nrows, ncols, 1)
    plt.plot(np.log(Fl_wage), Vf_W1[p.z_0 - 1, :, :])
    plt.title('W1')

    plt.subplot(nrows, ncols, 2)
    plt.plot(np.log(Fl_wage), Vf_J[0, :, :])
    plt.plot(np.log(Fl_wage), Vf_J[p.num_z-1, :, :])
    plt.title('J1')

    plt.subplot(nrows, ncols, 3)
    plt.plot(Vf_U)
    plt.title('U(X)')

    plt.subplot(nrows, ncols, 4)
    plt.plot(np.log(Fl_wage), Pr_e2u[2, :, :])
    plt.title('E2U')

    plt.subplot(nrows, ncols, 5)
    plt.plot(np.log(Fl_wage), Fl_ce[2, :, :])
    plt.title('Cost of effort')

    plt.show()


# Now, let's solve the full model ----------------------------------------------
js = JobSearchArray(p.num_x)
js.update(Vf_W1[p.z_0 - 1, :, :], prob_find_vx)

rho_grid = 1 / ru.utility_1d(p, w_grid) 
J1p = None
Fl_uf = ru.utility(p, unemp_bf)

# policies
rho_j2j = np.zeros((p.num_z, p.num_v, p.num_x))
rho_u2e = np.zeros((p.num_x))  # rho that worker gets when coming out of unemployment
rho_star = np.zeros((p.num_z, p.num_v, p.num_x)) #
qe_star  = np.zeros((p.num_z, p.num_v, p.num_x)) # quiting policy
pe_star  = np.zeros((p.num_z, p.num_v, p.num_x)) # job search policy
ve_star  = np.zeros((p.num_z, p.num_v, p.num_x)) # job search policy (value applied to)

# errors
error_w1  = 0
error_j   = 0
error_j1p = 0
error_js  = 0
niter     = 0

# Setting up the initial values for the VFI
Ui = Vf_U
Ji = Vf_J
W1i = Vf_W1

# create representation for J1p
start_time = time.time()

J1p = PowerFunctionGrid(W1i, Ji)

end_time = time.time()
execution_time = end_time - start_time
logger.info(f"Execution time: {execution_time} seconds")

gamma_all, rsqr = ru.fit_init_valuefunction(p, W1i, Ji)

EW1_star = Vf_J # ?
EJ1_star = Vf_J 

print(Vf_W1.shape)
print(Vf_J.shape)
print(Vf_J[1:5,1:5,1])
print(Vf_W1[1:5,1:5,1])

rho_bar = np.zeros((p.num_z, p.num_x))
rho_star = np.zeros((p.num_z, p.num_v, p.num_x))

# prepare expectation call
Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, Z_trans_mat.shape, X_trans_mat.shape)
Ex = oe.contract_expression('b,bx->x', Ui.shape, X_trans_mat.shape)
log_diff = np.zeros_like(EW1_star)

ite_num = 0
error_js = 1

update_eq = True


for ite_num in range(p.max_iter):
    # Store temporary value of J
    Ji2 = Ji
    W1i2 = W1i
    Ui2 = Ui

    # evaluate J1 tomorrow using our approximation
    Jpi = J1p.eval_at_W1(W1i)

    # we compute the expected value next period by applying the transition rules
    EW1i = Exz(W1i, Z_trans_mat, X_trans_mat)
    EJpi = Exz(Jpi, Z_trans_mat, X_trans_mat)
    EUi = Ex(Ui, X_trans_mat)

    # get worker decisions
    _, _, _, pc = ru.getWorkerDecisions(js, p, EW1i, EUi)

    print(pc.shape)
    print(pc[1:5,1:5,1])
    hi
    # get worker decisions at EW1i + epsilon
    _, _, _, pc_d = ru.getWorkerDecisions(js, p, EW1i + deriv_eps, EUi)

    # compute derivative where continuation probability is >0
    log_diff[:] = np.nan
    log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0])
    foc = rho_grid[ax, :, ax] - EJpi * log_diff / deriv_eps
    assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"

    for ix in range(p.num_x):
        for iz in range(p.num_z):

            assert np.all(EW1i[iz, 1:, ix] > EW1i[iz, :-1, ix])
            # find highest V with J2J search
            rho_bar[iz, ix] = np.interp(js.jsa[ix].e0, EW1i[iz, :, ix], rho_grid)
            rho_min = rho_grid[pc[iz, :, ix] > 0].min()  # lowest promised rho with continuation > 0

            # look for FOC below  rho_0
            Isearch = (rho_grid <= rho_bar[iz, ix]) & (pc[iz, :, ix] > 0)
            if Isearch.sum() > 0:
                rho_star[iz, Isearch, ix] = np.interp(rho_grid[Isearch],
                                                        ru.impose_increasing(foc[iz, Isearch, ix]),
                                                        rho_grid[Isearch], right=rho_bar[iz, ix])

            # look for FOC above rho_0
            Ieffort = (rho_grid > rho_bar[iz, ix]) & (pc[iz, :, ix] > 0)
            if Ieffort.sum() > 0:
                #assert np.all(foc[iz, Ieffort, ix][1:] > foc[iz, Ieffort, ix][:-1])
                rho_star[iz, Ieffort, ix] = np.interp(rho_grid[Ieffort],
                                                        foc[iz, Ieffort, ix], rho_grid[Ieffort])

            # set rho for quits to the lowest value
            Iquit = ~(pc[iz, :, ix] > 0)
            if Iquit.sum() > 0:
                rho_star[iz, Iquit, ix] = rho_min

            # get EW1_Star and EJ1_star
            EW1_star[iz, :, ix] = np.interp(rho_star[iz, :, ix], rho_grid, EW1i[iz, :, ix])
            EJ1_star[iz, :, ix] = np.interp(rho_star[iz, :, ix], rho_grid, EJpi[iz, :, ix])

    assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"
    
    # get pstar, qstar
    pe_star, re_star, qi_star, _ = ru.getWorkerDecisions(js, p, EW1_star, EUi)

    # Update firm value function
    Ji = fun_prod[:, ax, :] - w_grid[ax, :, ax] + p.beta * (1 - pe_star) * (1 - qi_star) * EJ1_star

    # Update worker value function
    W1i = ru.utility(p, w_grid)[ax, :, ax] - ru.effort_cost(p, qi_star) + \
        p.beta * qi_star * EUi + p.beta * (1-qi_star) * (re_star + EW1_star)
    W1i = .2*W1i + .8*W1i2

    # Update present value of unemployment
    if update_eq:
        _, rus, _, _ = ru.getWorkerDecisions(js, p, EUi, EUi, employed=False)
        Ui = ru.utility_gross(unemp_bf) + p.beta * (rus + EUi)
        Ui = 0.2*Ui + 0.8*Ui2

    # Updating J1 representation
    error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i, Ji)

    # Compute convergence criteria
    error_j1i = ru.array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
    error_j1g = ru.array_exp_dist(Jpi,J1p.eval_at_W1(W1i), 100)
    error_w1 = ru.array_dist(W1i, W1i2)
    error_u = ru.array_dist(Ui,Ui2)

    if (ite_num % 10 ==0):
        logger.debug('[{}]'.format(ite_num))

    # update worker search decisions
    if (ite_num % 10) == 0:
        if update_eq:
            # -----  check for termination ------
            if (np.array([error_w1, error_js, error_j1p_chg]).max() < p.tol_full_model
                    and ite_num > 50):
                break
            # ------ or update search function parameter using relaxation ------
            else:
                    P_xv = ru.matching_function(J1p.eval_at_W1(W1i)[p.z_0 - 1, :, :])
                    relax = 1 - np.power(1/(1+np.maximum(0,ite_num-p.eq_relax_margin)), p.eq_relax_power)
                    error_js = js.update(W1i[p.z_0 - 1, :, :], P_xv, type=1, relax=relax)
        else:
            # -----  check for termination ------
            if (np.array([error_w1, error_j1g]).max() < p.tol_full_model
                    and ite_num > 50):
                break

    if (ite_num % 25) == 0:
        logger.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e} U= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, error_u, js.rsq(), rsq_j1p ))
        
# --------- wrapping up the model ---------


# solve model
#logging.info("solving the model")
#model = wd.FullModel(p).solve(plot=False)

