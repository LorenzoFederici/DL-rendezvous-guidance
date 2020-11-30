from gym.envs.registration import register

register(
    id = 'rendezvous-v1', #this variable is what we pass into gym.make() to call our environment.
    entry_point = 'gym_rendezvous.envs:RendezvousEnv',
    kwargs = {'obs_type' : 0, \
        'random_obs' : False, 'randomIC' : False, 'stochastic' : False, \
        'termination' : True, 'NSTEPS' : 30., 'NITER' : 8e6, \
        'eps_schedule' : [1e-4], 'lambda_term' : 1., 'lambda_los' : 10., \
        'Dvmax' : 0.005, \
        'tf' : 6000., 'omega' : 8.5193e-4, \
        'rKOZ' : 0., 'beta_cone' : 180., \
        'r0' : [0.0, -15.0, 0.0], \
        'v0' : [0.0, 0.0, 0.0], \
        'rTf' : [0.0, -1.0, 0.0], \
        'vTf' : [0.0, 0.0, 0.0], \
        'dr0_max' : [0.0, 0.0, 0.0], 'dv0_max' : [0.0, 0.0, 0.0], \
        'sigma_r' : 0., 'sigma_v' : 0., \
        'sigma_u_rot' : 0., 'sigma_u_norm' : 0., \
        'MTE' : False, 'pr_MTE' : 0., 'max_MTE' : 0, \
        'acc_max' : [0.0, 0.0, 0.0]}
)