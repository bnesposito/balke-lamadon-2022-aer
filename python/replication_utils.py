import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
from scipy.stats import lognorm as lnorm
from scipy.optimize import minimize


ax = np.newaxis


def impose_decreasing(M):
    nv = M.shape[1]
    for v in reversed(range(nv-1)):
        M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
        #M = M.at[:,v,:].set(jnp.maximum(M[:,v,:],M[:,v+1,:]))
    return M


def impose_increasing(A0):
    A = np.copy(A0)
    nv = len(A)
    for v in range(1,nv):
        A[v] = np.maximum(A[v],A[v-1])
    return A


def construct_x_grid(p):
    """
        Construct a grid for worker productivity heterogeneity.
    """
    num_x0 = np.array(p.num_x / p.num_np, int)

    # the fixed heterogeneity component
    x0 = lnorm.ppf(q=np.linspace(0, 1, num_x0 + 2)[1:-1],
                    s=p.prod_var_x)
    # the time varyinng heterogeneity component
    xt = lnorm.ppf(q=np.linspace(0, 1, p.num_np + 2)[1:-1],
                    s=p.prod_var_x2)

    xx = np.kron(x0,xt) # permanent is slow moving
    xx = xx[ax,:] + np.zeros((p.num_z,p.num_x))
    return xx


def construct_z_grid(p):
    """
        Construct a grid for match productivity heterogeneity.
    """

    exp_z = np.tile(np.linspace(0, 1, p.num_z + 2)[1:-1][:, ax], (1, p.num_x))

    return lnorm.ppf(q=exp_z, s=p.prod_var_z)


def construct_v_array(p, unemp_bf, fun_prod):
        """
            Construct a grid for the value function using the production function to determine the min and max values.
            :return: An array of values corresponding to the value function realizations.
        """
        v_min = utility(p, np.min(unemp_bf)) / p.int_rate
        v_max = utility(p, 1.0 * np.max(fun_prod)) / (1 - p.beta)
        return np.linspace(v_min, v_max, p.num_v)

@jit
def utility(p, wage):
    """
        Computes the utility function at a particular wage.
        :param wage: Argument of the function.
        :return: Output of the function.
    """
    aa = p.u_a * jnp.power(p.tax_tau, 1 - p.u_rho) 
    return jnp.divide(aa * jnp.power(wage, p.tax_lambda * (1.0 - p.u_rho)) - p.u_b,
                        1 - p.u_rho)
    # return np.divide(self.p.u_a * np.power(wage, 1 - self.p.u_rho) - self.p.u_b,
    #                  1 - self.p.u_rho)

@jit
def utility_gross(p, wage):
    """
        Computes the utility function at a particular wage, not applying the tax function
        :param wage: Argument of the function.
        :return: Output of the function.
    """
    return jnp.divide(p.u_a * jnp.power(wage, 1 - p.u_rho) - p.u_b,
                        1 - p.u_rho)

@jit
def inv_utility(p, value):
    """
        Computes the inverse utility function at a particular value.
        :param value: Argument of the function.
        :return: Output of the function.
    """
    aa = p.u_a * jnp.power(p.tax_tau, 1.0 - p.u_rho) 
    return jnp.power(jnp.divide((1.0 - p.u_rho) * value + p.u_b, aa),
                    (jnp.divide(1.0, p.tax_lambda * (1.0 - p.u_rho))))

@jit
def utility_1d(p, wage):
    """
        Computes the first derivative of the utility function at a particular wage.
        :param wage: Argument of the function.
        :return: Output of the function.
    """
    #return self.p.u_a * np.power(wage, - self.p.u_rho)
    aa = p.u_a * jnp.power(p.tax_tau, 1.0 - p.u_rho) 
    return aa * p.tax_lambda * jnp.power(wage, p.tax_lambda * ( 1.0 - p.u_rho) - 1.0)

@jit
def inv_utility_1d(p, value):
    """
        Computes the first derivative of the inverse utility function at a particular value.
        :param value: Argument of the function.
        :return: Output of the function.
    """
    aa = p.u_a * jnp.power(p.tax_tau, 1 - p.u_rho) 
    pow_arg = ( (1 - p.u_rho) * value + p.u_b   ) / aa
    return jnp.power( pow_arg, 1.0/(p.tax_lambda * (1 - p.u_rho) ) - 1.0) / ( p.tax_lambda * aa )

@jit
def effort_cost(p, q_value):
    """
        Computes the effort cost function given the level of effort 'q'.
        :param q_value: Argument of the function.
        :return: Output of the function.
    """
    gam = jnp.power(p.efcost_ce, -1)
    return p.efcost_sep * q_value + \
        jnp.divide(p.efcost_sep,  p.efcost_ce - 1) - \
        jnp.divide(p.efcost_sep, 1 - gam) * jnp.power(q_value,(1 - gam))


def inv_effort_cost_1d(p, V):
    """
        Returns the quit probability by computing the inverse of the first derivative of the
        effort cost function.
        :param eff: Argument of the function.
        :return: Output of the function.
    """
    #return np.power(1 + np.divide(np.maximum(eff, 0), self.p.efcost_sep), -self.p.efcost_ce)
    return np.power(1 + np.divide(np.maximum(-V, 0), p.efcost_sep), - p.efcost_ce)

@jit
def log_consumption_eq(p, V):
    """
        Returns the log wage/consumption equivalent associated with a present value of the worker.
    """
    return(jnp.log(inv_utility( (1-p.beta) * V )))

@jit
def log_profit_eq(p, J):
    """
        Returns the log profit equivalent associated with the firm present value
    """
    return( jnp.log( (1-p.beta) * J))

@jit
def consumption_eq(p, V):
    """
        Returns the log wage/consumption equivalent associated with a present value of the worker.
    """
    return((inv_utility( (1-p.beta) * V )))

@jit
def profit_eq(p, J):
    """
        Returns the log profit equivalent associated with the firm present value
    """
    return(( (1-p.beta) * J))

@jit
def matching_function(p, J1):
    return p.alpha * jnp.power(1 - jnp.power(
        jnp.divide(p.kappa, jnp.maximum(J1, p.kappa)), p.sigma), 1 / p.sigma)


# Functions from valuefunction
def curve_fit_search_and_grad(gamma, Xi, Yi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/ 100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    Ri     = gamma[0] + gamma[1] * Xi_pow - Yi
    val    = np.power(Ri, 2).mean()

    # the optimizer can handle invalid returns for gradient
    # with np.errstate(divide='ignore'):
    #     with np.errstate(invalid='ignore'):
    g1     = 2 * Ri.mean()
    g2     = 2 * ( Ri * Xi_pow ).mean()
    g3     = 2 * ( Ri * np.log( Xi_arg ) * Xi_pow * gamma[1] ).mean()
    g4     = 2 * ( Ri * gamma[1] * gamma[2] * np.exp(gamma[3]) * np.power( Xi_arg , gamma[2] - 1 ) ).mean()

    return val, np.array([g1,g2,g3,g4])


def fit_init_valuefunction(p, W1, J1, weight=0.01):

    gamma_all = np.zeros( (p.num_z, p.num_x,5) )
    rsqr  = np.zeros( (p.num_z, p.num_x))
    
    # we fit for each (z,x)
    p0 = [0, -1, -1, np.log(0.1)]
    for ix in range(p.num_x):
        for iz in range(p.num_z):
            p0[0] = J1[iz, 0, ix]
            res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                            options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                            args=(W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max()))
            p0 = res2.x
            gamma_all[iz, ix, 0:4] = res2.x
            gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
            rsqr[iz, ix] = res2.fun / np.power(J1[iz, :, ix],2).mean()
    
    return gamma_all, rsqr


# Functions from modelfull
def getWorkerDecisions(js, p, EW1, EU, employed=True):
        """
        :param EW1: Expected value of employment
        :param EU:  Expected value of unemployment
        :param employed: whether the worker is employed (in which case we multiply by efficiency
        :return: pe,re,qi search decision and associated return, as well as quit decision.
        """
        pe, re = js.solve_search_choice(EW1)
        print((~np.isnan(EW1)).sum(), (~np.isinf(EW1)).sum())
        print((~np.isnan(pe)).sum(), (~np.isinf(pe)).sum())
        print((~np.isnan(re)).sum(), (~np.isinf(re)).sum())
        assert (~np.isnan(pe)).all(), "pe is not NaN"
        assert (pe <= 1).all(), "pe is not less than 1"
        assert (pe >= -1e-10).all(), "pe is not larger than 0"

        if employed:
            pe = pe * p.s_job
            re = re * p.s_job

        # solve quit effort
        # qi = self.pref.inv_effort_cost_1d(self.p.beta * (re + (W1i - Ui) + 1/rho_grid[ax,:,ax] * Ji))
        V = - p.beta * (re + EW1 - EU)
        print(V[1:5,1:5,1])
        print((~np.isnan(V)).sum(), (~np.isinf(V)).sum())
        hi
        np.power(1 + np.divide(np.maximum(-V, 0), p.efcost_sep), - p.efcost_ce)
        qi = inv_effort_cost_1d(p, - p.beta * (re + EW1 - EU))
        #assert (qi <= 1).all(), "qi is not less than 1"
        #assert (qi >= 0).all(), "qi is not larger than 0"

        # construct the continuation probability
        pc = (1 - pe) * (1 - qi)

        return pe, re, qi, pc


def array_exp_dist(A,B,h):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    # log_weight = - 0.5*np.power(B/h,2) 
    # # handling underflow gracefully
    # log_weight = log_weight - log_weight.max()
    # weight = np.exp( np.maximum( log_weight, -100))
    # return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 
    weight = np.exp( - 0.5*np.power(B/h,2))
    return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 



def array_dist(A,B):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    return  (np.power( A-B,2) ).mean() / ( np.power(B,2) ).mean() 