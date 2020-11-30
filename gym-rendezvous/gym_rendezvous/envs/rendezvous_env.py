# Filter tensorflow version warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines.common import set_global_seeds

import math
import numpy as np
from numpy import sqrt, log, exp, cos, sin, tan, arccos, cross, dot, array
from numpy.linalg import norm
from numpy.random import uniform as np_uniform
from numpy.random import normal as np_normal
from numpy.random import randint as np_randint

from gym_rendezvous.envs.pyHCW import propagate_HCW, par2ic, ic2par


""" RL SPACE RENDEZVOUS ENVIRONMENT CLASS """
class RendezvousEnv(gym.Env):
    """
    Reinforcement Learning environment,
    for a minimum-Dv, time-fixed, impulsive linear rendezvous maneuver 
    with fixed initial and final relative conditions.
    The aim of the mission is to come in close proximity of a target body.
    Two type of operational constraints are considered: a spherical keep-out zone, 
    representing the space occupied by the target, and a visibility cone 
    the chaser must move through during the final part of the maneuver.
    The chaser trajectory is divided in NSTEPS segments, and 
    the Hill-Clohessy-Wiltshire (HCW) equations are used to propagate the chaser relative
    motion between any two impulsive Dvs. 
    The environment can be deterministic or stochastic (i.e., with state,
    observation and/or control uncertainties, as well as with
    stochastic dynamical perturbations).
    The initial state of the chaser can be fixed or uniformly scattered.

    Class inputs:
        - (int) obs_type: 0 = observations are relative chaser position and 
                            velocity in RTN;
                          1 = observations are chaser position and 
                            velocity in RTN + current action;
        - (bool) random_obs: False = deterministic observations, 
                             True = random observations
        - (bool) randomIC:   True = random initial conditions,
                             False = fixed initial conditions
        - (bool) stochastic: True = stochastic environment,
                             False = deterministic environment
        - (bool) termination: True = terminate episode when constaints violated
                              False = otherwise
        - (int) NSTEPS: number of trajectory segments
        - (inr) NITER: number of training iterations
        - (list) eps_schedule: list of tolerance values for constraint satisfaction
        - (float) lambda_term: weight of terminal constraint violation in reward
        - (float) lambda_los: weight of visibility cone (los) constraint violation in reward
        - (float) DVmax: maximum engine DV, km/s
        - (float) omega: mean motion of the target body, rad/s
        - (float) rKOZ: radius of the keep out zone, km
        - (float) beta_cone: visibility cone semi-aperture, rad
        - (float) tf: total mission time, s
        - (list) r0: initial relative chaser position, km
        - (list) v0: initial relative chaser velocity, km/s
        - (list) rTf: final (target) relative chaser position, km
        - (list) vTf: final (target) relative chaser velocity, km/s
        - (list) dr0_max: maximum variation of initial position, km
        - (list) dv0_max: maximum variation of initial velocity, km/s
        - (float) sigma_r: standard deviation on position, km
        - (float) sigma_v: standard deviation on velocity, km/s
        - (float) sigma_u_rot : standard deviation on control vector rotation, rad
        - (float) sigma_u_norm : standard deviation on control vector modulus (percentage)
        - (bool) MTE: at least one missed thrust events (MTE) occurs?
        - (float) pr_MTE: probability of having a new MTE after the last one
        - (int) max_MTE: maximum number of consecutive MTEs
        - (list) acc_max: maximum value of the components of a perturbing acceleration, km/s^2

    RL ENVIRONMENT (MDP)
    Observations: 
        Type: Box(7)
        if (obs_type == 0):
        Num	Observation                Min     Max
        0	x                        -max_r   max_r
        1	y                        -max_r   max_r
        2	z                        -max_r   max_r
        3	xdot                     -max_v   max_v
        4	ydot                     -max_v   max_v
        5	zdot                     -max_v   max_v
        6   N - k                       0       N
        Type: Box(10)
        else if (obs_type == 1):
        Num	Observation                Min     Max
        0	x                        -max_r   max_r
        1	y                        -max_r   max_r
        2	z                        -max_r   max_r
        3	xdot                     -max_v   max_v
        4	ydot                     -max_v   max_v
        5	zdot                     -max_v   max_v
        6   N - k                       0       N
        7   Dvx                        -1       1
        8   Dvy                        -1       1
        9   Dvz                        -1       1
        
    Actions:
        Type: Box(3)
        Num	Action
        0	Dvx                        -1       1
        1	Dvy                        -1       1
        2	Dvz                        -1       1

    Reward:
        [at any time step]     -norm(Dv)
        [at final time]        -norm(r-rTf)
        [at final time]        -norm(v-vTf)

    Starting State:
        Start at state: (r0, v0)
    
    Episode Termination:
        - At time tf
        - Position and/or velocity reach the environment boundaries

    """

    metadata = {'render.modes': ['human']}

    """ Class constructor """
    def __init__(self, obs_type, \
        random_obs, randomIC, stochastic, \
        termination, NSTEPS, NITER, \
        eps_schedule, lambda_term, lambda_los ,\
        Dvmax, tf, omega, \
        rKOZ, beta_cone, \
        r0, v0, \
        rTf, vTf, \
        dr0_max, dv0_max, \
        sigma_r, sigma_v, \
        sigma_u_rot, sigma_u_norm, \
        MTE, pr_MTE, max_MTE, acc_max):

        super(RendezvousEnv, self).__init__()

        """ Class attributes """
        self.obs_type = int(obs_type)
        self.random_obs = bool(random_obs)
        self.randomIC = bool(randomIC)
        self.stochastic = bool(stochastic)
        self.termination = bool(termination)
        self.NSTEPS = float(NSTEPS)
        self.NITER = float(NITER)
        self.eps_schedule = array(eps_schedule)
        self.lambda_term = float(lambda_term)
        self.lambda_los = float(lambda_los)
        self.Dvmax = float(Dvmax)
        self.tf = float(tf)
        self.omega = float(omega)
        self.rKOZ = float(rKOZ)
        self.beta_cone = float(beta_cone)
        self.r0 = array(r0)
        self.v0 = array(v0)
        self.rTf = array(rTf)
        self.vTf = array(vTf)
        self.dr0_max = array(dr0_max)
        self.dv0_max = array(dv0_max)
        self.sigma_r = float(sigma_r)
        self.sigma_v = float(sigma_v)
        self.sigma_u_rot = float(sigma_u_rot)
        self.sigma_u_norm = float(sigma_u_norm)
        self.MTE = bool(MTE)              
        self.pr_MTE = float(pr_MTE)
        self.max_MTE = int(max_MTE)
        self.acc_max = array(acc_max)
        
        """ Time variables """
        self.dt = self.tf / self.NSTEPS     # time step, s
        self.training_steps = 0.            # number of training steps

        """ Normalization quantities """
        self.rconv = max(norm(self.r0), norm(self.rTf))
        self.vconv = max([norm(self.v0), norm(self.vTf), self.Dvmax])
        self.tconv = self.tf

        """ Environment boundaries """
        self.max_r = 2.                                     # maximum distance, adim
        self.max_v = 20.                                    # maximum velocity, adim
        self.max_Dvtot = self.NSTEPS*self.Dvmax             # maximum total Dv, km/s

        """ Deterministic environment """
        self.deterministic = (self.stochastic == False and self.random_obs == False and self.randomIC == False)
        
        """ OBSERVATION SPACE """
        if self.obs_type == 0:
            # Lower bounds
            x_lb = np.array([-self.max_r, -self.max_r, -self.max_r, \
                -self.max_v, -self.max_v, -self.max_v, \
                0.])
            # Upper bounds
            x_ub = np.array([+self.max_r, +self.max_r, +self.max_r, \
                +self.max_v, +self.max_v, +self.max_v, \
                self.NSTEPS])
        else:
            # Lower bounds
            x_lb = np.array([-self.max_r, -self.max_r, -self.max_r, \
                -self.max_v, -self.max_v, -self.max_v, \
                0., -1., -1., -1.])
            # Upper bounds
            x_ub = np.array([+self.max_r, +self.max_r, +self.max_r, \
                +self.max_v, +self.max_v, +self.max_v, \
                self.NSTEPS, 1., 1., 1.])
        
        self.observation_space = spaces.Box(x_lb, x_ub, dtype=np.float64)

        """ ACTION ASPACE """
        # Lower bounds
        a_lb = np.array([-1., -1., -1.])
        # Upper bounds
        a_ub = np.array([1., 1., 1.])

        self.action_space = spaces.Box(a_lb, a_ub, dtype=np.float64)
        
        """ Environment initialization """
        self.viewer = None
        self.state = None
    
    """ Set seed """
    def seed(self, seed=None):
        """
        :return seed: current seed of pseudorandom
            numbers generator
        """
        set_global_seeds(seed)
        
        return [seed]

    """ Get Reward """
    def getReward(self, done, rk, vk, action):
        """
        :param done: episode is terminated?
        :param rk: current position
        :param vk: current velocity
        :param action: current action
        :return reward: current reward
        """
        # Constraint satisfaction tolerance
        eps = self.eps_constraint()

        # Frequent reward: current action norm
        reward = - (norm(action)*self.Dvmax)/self.max_Dvtot

        # Penalty: current action greater than maximum admissible
        reward -= 100.*max(0., norm(action) - 1.)

        #Delayed reward: final constraint violation
        if done: 
            
            # Constraint violation on final position
            r_viol = norm(rk - self.rTf)

            #Final burn
            Dvf = self.vTf - vk
            
            #DV norm
            reward -= norm(Dvf)/self.max_Dvtot

            # Constraint violation on final velocity
            v_viol = max(0., norm(Dvf) - self.Dvmax)*100.

            #Violation
            c_viol = max(0., max(r_viol, v_viol) - eps)
            reward -= self.lambda_term*c_viol

        # Additional penality when operative constraints violated
        violated, max_viol = self.operConstraints(rk)
        if violated and not self.termination:
            reward -= self.lambda_los*max(0., max_viol - eps)
                
        return reward
    
    """ Episode termination """
    def operConstraints(self, rk):
        """
        :param rk: current position
        :return violated: are KOZ and/or visibility cone
            constraints violated?
        :return max_viol: maximum constraint violation

        """
        violated = 0
        max_viol = 0.
       
        # Keep Out Zone
        r_chaser = norm(rk)
        if (r_chaser < self.rKOZ):
            # Chaser position
            x = rk[0]
            y = rk[1]
            z = rk[2]

            #Pyramidal visibility cone constraints
            dis1 = -x + y*tan(self.beta_cone)
            dis2 = x + y*tan(self.beta_cone)
            dis3 = z + y*tan(self.beta_cone)
            dis4 = -z + y*tan(self.beta_cone)

            max_viol = max(0., max([dis1, dis2, dis3, dis4]))
            if max_viol > 0.:
                violated = 1

        return bool(violated), max_viol

    """ Episode termination """
    def isDone(self, tk, rk):
        """
        :param tk: current time
        :param rk: current position
        :return bool: terminate the episode?

        """
       
        if tk == self.NSTEPS:
            return True
        
        if self.termination:
            violated, max_viol = self.operConstraints(rk)
            if violated:
                return True

        return False

    """ Safe Episode termination """
    def safeStop(self, r, v):
        """
        :param r: current position, km
        :param v: current velocity, km/s
        :return bool: terminate the episode?

        """

        if (norm(r)/self.rconv > self.max_r):
            return True
        elif (norm(v)/self.vconv > self.max_v):
            return True
        else:
            return False
    
    """ Get epsilon value """
    def eps_constraint(self):
        """
        :return epsilon: epsilon value at current training step,
            given eps_schedule, decreasing with a piecewise constant schedule

        """
        
        n_values = len(self.eps_schedule)
        n_steps_per_value = self.NITER / n_values

        for i in range(n_values):
            if (self.training_steps <= (i+1)*n_steps_per_value):
                return self.eps_schedule[i]

        return self.eps_schedule[n_values-1]
    
    """ Initial conditions variation"""
    def ICvariation(self):
        """
        :return dr0, dv0: variation on position (km), and
            velocity (km/s) wrt to the given initial conditions

        """

        #Position error
        dx0 = np_uniform(-self.dr0_max[0], self.dr0_max[0])
        dy0 = np_uniform(-self.dr0_max[1], self.dr0_max[1])
        dz0 = np_uniform(-self.dr0_max[2], self.dr0_max[2])
        dr0 = array([dx0, dy0, dz0])

        #Velocity error
        dxdot0 = np_uniform(-self.dv0_max[0], self.dv0_max[0])
        dydot0 = np_uniform(-self.dv0_max[1], self.dv0_max[1])
        dzdot0 = np_uniform(-self.dv0_max[2], self.dv0_max[2])
        dv0 = array([dxdot0, dydot0, dzdot0])

        return dr0, dv0
    
    """ Get observation errors at step tk """
    def obsErrors(self):
        """
        :return drk, dvk: errors on position (km), and
            velocity (km/s) at step tk

        """

        #Position error
        drk = np_normal(0., self.sigma_r, 3)

        #Velocity error
        dvk = np_normal(0., self.sigma_v, 3)

        return drk, dvk
    
    """ Get state errors at step tk """
    def stateErrors(self):
        """
        :return drk, dvk: errors on position (km), and
            velocity (km/s) at step tk

        """

        #Position error
        drk = np_normal(0., self.sigma_r, 3)

        #Velocity error
        dvk = np_normal(0., self.sigma_v, 3)

        return drk, dvk
    
    """ Perturbate the action at step tk """
    def perturbAction(self, action):
        """
        :param action: current action
        :return action_pert: perturbed action

        """
        #MTE
        if self.MTE == True:
            if (self.tk == self.tk_mte) and (self.n_mte < self.max_MTE):
                self.n_mte += 1
                pr = np_uniform(0., 1.)
                if (pr < self.pr_MTE) and (self.tk < self.NSTEPS - 2):
                    self.tk_mte += 1
                return array([0., 0., 0.])

        #Modulus error
        du_norm = np_normal(1., self.sigma_u_norm)
        np.clip(du_norm, 1. - 10.*self.sigma_u_norm, 1. + 10.*self.sigma_u_norm)

        #Rotation error
        du_rot = np_normal(0., self.sigma_u_rot, 3)
        np.clip(du_rot, -3.*self.sigma_u_rot, 3.*self.sigma_u_rot)

        #Rotation matrix
        Arot = array([[1., -du_rot[2], du_rot[1]], 
                  [du_rot[2], 1., -du_rot[0]], 
                  [-du_rot[1], du_rot[0], 1.]])

        #Perturbed action
        action_pert = du_norm*(Arot.dot(action))

        return action_pert
    
    """ Perturbing acceleration at step tk """
    def perturbAcceleration(self):
        """
        :return acc: perturbing acceleration
        """
        acc_x = np_uniform(-self.acc_max[0], self.acc_max[0])
        acc_y = np_uniform(-self.acc_max[1], self.acc_max[1])
        acc_z = np_uniform(-self.acc_max[2], self.acc_max[2])
        acc = array([acc_x, acc_y, acc_z])

        return acc

    """ Propagation step """
    def propagation_step(self, rk, vkm, Dvk, dt):
        """
        :param rk: current position
        :param vkm: velocity before Dv
        :param Dvk: Dv
        :param dt: time of flight
        :return uk: control at the current time step
        :return rk1: position at the beginning of next time step
        :return vk1m: velocity at the beginning of next time step

        """
        
        # Velocity after DV
        vkp = vkm + Dvk

        # Position and velocity at the next time step
        rk1, vk1m = propagate_HCW(r0 = rk, v0 = vkp, dt = dt, omega = self.omega)          

        return rk1, vk1m
    
    """ Return observations """
    def eval_observations(self, rk_obs, vk_obs, tk_left, action):
        """
        :param rk_obs: observation of current position
        :param vk_obs: observation of current velocity
        :param tk_left: time-steps to go
        :param action: current action
        :return obs: current observation

        """
        # Observations
        if self.obs_type == 0:
            obs = np.array([rk_obs[0]/self.rconv, rk_obs[1]/self.rconv, rk_obs[2]/self.rconv, \
                vk_obs[0]/self.vconv, vk_obs[1]/self.vconv, vk_obs[2]/self.vconv, \
                tk_left]).astype(np.float64)
        else:
            obs = np.array([rk_obs[0]/self.rconv, rk_obs[1]/self.rconv, rk_obs[2]/self.rconv, \
                vk_obs[0]/self.vconv, vk_obs[1]/self.vconv, vk_obs[2]/self.vconv, \
                tk_left, action[0], action[1], action[2]]).astype(np.float64)
        
        return obs

    """ Do forward step in the MDP """
    def step(self, action):
        """
        :param action: current action
        :return obs, reward, done, info

        """
        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Perturbate the action
        if self.stochastic == True:
            #Control error
            action = self.perturbAction(action)
            action = min(sqrt(3.), norm(action))*action/norm(action)

            #Perturbing acceleration
            acc = self.perturbAcceleration()
        else:
            acc = np.array([0., 0., 0.])

        # Dv
        Dvk = action*self.Dvmax

        # State at next time step and current control
        rk1, vk1m = self.propagation_step(self.rk, self.vkm, Dvk+acc*self.dt, self.dt)

        # Perturbate the state
        if self.stochastic == True and self.tk < self.NSTEPS-1:
            drk, dvk = self.stateErrors()
            rk1 = rk1 + drk
            vk1m = vk1m + dvk

        # Info (state at the beginning of the segment)
        self.sol['rx'] = self.rk[0]
        self.sol['ry'] = self.rk[1]
        self.sol['rz'] = self.rk[2]
        self.sol['vx'] = self.vkm[0]
        self.sol['vy'] = self.vkm[1]
        self.sol['vz'] = self.vkm[2]
        self.sol['ux'] = Dvk[0]
        self.sol['uy'] = Dvk[1]
        self.sol['uz'] = Dvk[2]
        self.sol['t'] = self.t
        info = self.sol
            
        # Update the spacecraft state
        self.rk = rk1
        self.vkm = vk1m
        self.tk += 1.
        self.tk_left -= 1.
        self.t = self.tk*self.dt
        self.t_left = self.tk_left*self.dt           

        #Errors on observations
        rk_obs = self.rk
        vkm_obs = self.vkm
        if self.random_obs == True:
            drk, dvk = self.obsErrors()
            rk_obs = rk_obs + drk
            vkm_obs = vkm_obs + dvk

        # Observations
        obs = self.eval_observations(rk_obs, vkm_obs, self.tk_left, action)
        
        # Update training steps
        self.training_steps += 1.

        # Episode termination
        done = (self.isDone(self.tk, self.rk) or self.safeStop(self.rk, self.vkm))

        # Reward
        reward = self.getReward(done, self.rk, self.vkm, action)

        # Penalty outside boundaries
        if self.safeStop(self.rk, self.vkm):
            reward = reward - 1000.*self.tk_left

        return obs, float(reward), done, info

    """ Initialize the episode """
    def reset(self):
        """
        :return obs: observation vector

        """

        # Environment variables
        self.rk = self.r0                   # position at the k-th time step, km
        self.vkm = self.v0                  # velocity at the k-th time step, before Dv, km/s
        self.tk = 0.                        # k-th time step
        self.tk_left = self.NSTEPS          # steps-to-go
        self.t = 0.                         # time from departure
        self.t_left = self.tf               # time-to-go

        # Random initial conditions
        if (self.randomIC == True):
            dr0, dv0 = self.ICvariation()
            self.rk = self.rk + dr0
            self.vkm = self.vkm + dv0
        
        if (self.stochastic == True):
            #Select the segment with MTE
            pr = np_uniform(0., 1.)
            if (self.MTE == True) and (pr < 1):
                self.tk_mte = np_randint(0, int(self.NSTEPS))
                self.n_mte = 0 #number of MTEs so far
            else:
                self.tk_mte = -1

        # Reset parameters
        self.sol = {'rx': [], 'ry': [], 'rz': [],
                    'vx': [], 'vy': [], 'vz': [],
                    'ux': [], 'uy': [], 'uz': [],
                    't': []}
        self.done = False

        rk_obs = self.rk
        vkm_obs = self.vkm

        obs = self.eval_observations(rk_obs, vkm_obs, self.tk_left, [0., 0., 0.])

        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass