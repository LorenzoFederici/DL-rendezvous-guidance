import os
import platform
import sys
import warnings
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, DDPG, SAC, TD3, TRPO
from stable_baselines.common.callbacks import EvalCallback
from custom_modules.custom_policies import CustomPolicy_2x32, CustomPolicy_3x64, CustomPolicy_4x128, \
    CustomLSTMPolicy, CustomPolicy_2x81, CustomPolicy_3_var
from custom_modules.learning_schedules import linear_schedule
from custom_modules.env_fun import make_env
from custom_modules.plot_results import plot_results
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import time
import gym_rendezvous

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

if __name__ == '__main__':

    #Input data
    postprocess = True
    MonteCarlo = True
    tensorboard = False
    eval_environment = True

    #Input settings file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default="settingsRL.txt", \
        help='Input settings file')
    parser.add_argument('--envs', type=int, default=-1, \
        help='Number of environments')
    parser.add_argument('--input_model_folder', type=str, default="sol_saved/-1traj_8nb_500epochs_95frac_1/", \
        help='Folder of the input BC model to load')
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings
    input_model_folder = "./" + args.input_model_folder
    if os.path.isfile(input_model_folder + "best_model.zip"):
        input_model = input_model_folder + "best_model"
    else:
        input_model = input_model_folder + "final_model"

    #Read settings and assign environment and model parameters
    with open(settings_file, "r") as input_file: # with open context
        input_file_all = input_file.readlines()
        for line in input_file_all: #read line
            line = line.split()
            if (len(line) > 2):
                globals()[line[0]] = line[1:]
            else:
                globals()[line[0]] = line[1]
            
    input_file.close() #close file

    #Settings
    load_model = bool(int(load_model)) #load external model

    #Environment parameters
    obs_type = int(obs_type) #type of observations
    random_obs = bool(int(random_obs)) #random observations
    randomIC = bool(int(randomIC)) #random initial conditions
    stochastic = bool(int(stochastic)) #stochastic environment
    termination = bool(int(termination)) #terminating when constrainits violated
    NSTEPS = int(NSTEPS) # Total number of trajectory segments
    if isinstance(eps_schedule, list):
        eps_schedule = [float(i) for i in eps_schedule] #epsilon-constraint schedule
    else:
        eps_schedule = [float(eps_schedule)]
    lambda_term = float(lambda_term) #weight of terminal constraint violation in reward
    lambda_los = float(lambda_los) #weight of visibility cone constraint violation in reward
    dr0_max = [float(i) for i in dr0_max] #maximum variation of initial position
    dv0_max = [float(i) for i in dv0_max] #maximum variation of initial velocity
    sigma_r = float(sigma_r) #standard deviation of position
    sigma_v = float(sigma_v) #standard deviation of velocity
    sigma_u_rot = float(sigma_u_rot) #standard deviation of control rotation
    sigma_u_norm = float(sigma_u_norm) #standard deviation of control modulus
    MTE = bool(int(MTE)) #at least one MTE occurs?
    pr_MTE = float(pr_MTE) #probability of having a MTE at k-th step
    max_MTE = int(max_MTE) #maximum number of consecutive MTEs
    acc_max = [float(i) for i in acc_max] #maximum value of the components of a perturbing acceleration
    seed = 0 #int(time.time()) #pseudorandom number generator seed

    #Model parameters
    if load_model == True:
        policy = globals()[policy]
    num_cpu = int(num_cpu) #number of environments
    if (args.envs > 0):
        num_cpu = int(args.envs) #number of environments
    learning_rate_in = float(learning_rate_in) #initial learning rate
    if (algorithm == "PPO"):
        clip_range_in = float(clip_range_in) #initial clip range
    if learning_rate == "lin":
        learning_rate_type = "linear"
        learning_rate = linear_schedule(learning_rate_in)
    elif learning_rate == "const":
        learning_rate_type = "constant"
        learning_rate = learning_rate_in
    if (algorithm == "PPO"):
        if clip_range == "lin":
            clip_range = linear_schedule(clip_range_in)
        elif clip_range == "const":
            clip_range = clip_range_in
    if (algorithm == "PPO" or algorithm == "A2C" or \
        algorithm == "SAC" or algorithm == "TRPO"):
        ent_coef = float(ent_coef)
    gamma = float(gamma)
    if (algorithm == "PPO" or algorithm == "TRPO"):
        lam = float(lam)
    if (algorithm == "PPO"):
        noptepochs = int(noptepochs)
    nminibatches = int(nminibatches) #Number of batches
    niter = int(float(niter)) #number of iterations
    n_steps = int(NSTEPS*nminibatches) #batch size
    niter_per_cpu = niter / num_cpu #Steps per CPU

    #Output folders and log files
    if load_model == True:
        out_folder_root = input_model_folder + "RL/"
    else:
        out_folder_root = "./sol_saved/"
    n_sol = 1
    out_folder_root += str(num_cpu) + "env_" + \
        str(nminibatches) + "nb_" + \
        str(int(niter/1e6)) + "Mstep_"
    out_folder = out_folder_root + str(n_sol) + "/"
    while os.path.exists(out_folder):
        n_sol += 1
        out_folder = out_folder_root + str(n_sol) + "/"
    os.makedirs(out_folder, exist_ok=True)
    trained_model_name = "final_model"
    if tensorboard == True:
        tensorboard_log = out_folder
    else:
        tensorboard_log = None
    trained_model_log = out_folder + trained_model_name
    monitor_folder = out_folder + algorithm + "/"
    os.system("cp " + settings_file + " " + out_folder)
    os.system("mv " + out_folder + args.settings + " " + out_folder + "settings.txt")

    #Problem data
    Dvmax = float(Dvmax)                        #Maximum chaser Dv, km/s
    omega = float(omega)                        #Mean motion of the target, rad/s
    rKOZ = float(rKOZ)                          #radius of the keep out zone (KOZ), km
    beta_cone = float(beta_cone)*np.pi/180.   #visibility cone semi-aperture, rad

    #Read Mission file
    t_nom = []  #nominal trajectory: time
    rx_nom = [] #nominal trajectory: x
    ry_nom = [] #nominal trajectory: y
    rz_nom = [] #nominal trajectory: z
    vx_nom = [] #nominal trajectory: vx
    vy_nom = [] #nominal trajectory: vy
    vz_nom = [] #nominal trajectory: vz
    mission_folder = "missions/"
    mission_file = mission_folder + mission_name + ".dat" #File with mission data
    with open(mission_file, "r") as f: # with open context
        f.readline()
        file_all = f.readlines()
        for line in file_all: #read line
            line = line.split()
            state = np.array(line).astype(np.float64) #non-dimensional data
            
            #save data
            t_nom.append(state[0])
            rx_nom.append(state[1])
            ry_nom.append(state[2])
            rz_nom.append(state[3])
            vx_nom.append(state[4])
            vy_nom.append(state[5])
            vz_nom.append(state[6])

    f.close() #close file

    #Mission data

    #Time-of-flight
    tf =  t_nom[-1] - t_nom[0]   #s, Time-of-flight

    #Reference initial state
    r0 = [rx_nom[0], ry_nom[0], rz_nom[0]] #km, initial relative chaser position
    v0 = [vx_nom[0], vy_nom[0], vz_nom[0]] #km/s, initial relative chaser velocity

    #Target state
    rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]      #km, final target position
    vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]      #km/s, final target velocity
    
    # Create the environment

    #Create the vectorized environment
    for rank in range(num_cpu):
        os.makedirs(monitor_folder + "env_" + str(rank) + "/", exist_ok=True)

    if num_cpu <= 1:
        env = DummyVecEnv([make_env(env_id=env_name, rank=rank, seed=seed, \
            filename=monitor_folder + "env_" + str(rank) + "/", \
            obs_type=obs_type, \
            random_obs=random_obs, randomIC=randomIC, stochastic=stochastic, \
            termination=termination, NSTEPS=NSTEPS, NITER=niter_per_cpu, \
            eps_schedule=eps_schedule, lambda_term=lambda_term, \
            Dvmax=Dvmax, tf=tf, omega=omega, \
            rKOZ=rKOZ, beta_cone=beta_cone, \
            r0=r0, v0=v0, \
            rTf=rTf, vTf=vTf, \
            dr0_max=dr0_max, dv0_max=dv0_max, \
            sigma_r=sigma_r, sigma_v=sigma_v, \
            sigma_u_rot=sigma_u_rot, sigma_u_norm=sigma_u_norm, \
            MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE, \
            acc_max=acc_max) for rank in range(num_cpu)])
    else:
        env = SubprocVecEnv([make_env(env_id=env_name, rank=rank, seed=seed, \
            filename=monitor_folder + "env_" + str(rank) + "/", \
            obs_type=obs_type, \
            random_obs=random_obs, randomIC=randomIC, stochastic=stochastic, \
            termination=termination, NSTEPS=NSTEPS, NITER=niter_per_cpu, \
            eps_schedule=eps_schedule, lambda_term=lambda_term, \
            Dvmax=Dvmax, tf=tf, omega=omega, \
            rKOZ=rKOZ, beta_cone=beta_cone, \
            r0=r0, v0=v0, \
            rTf=rTf, vTf=vTf, \
            dr0_max=dr0_max, dv0_max=dv0_max, \
            sigma_r=sigma_r, sigma_v=sigma_v, \
            sigma_u_rot=sigma_u_rot, sigma_u_norm=sigma_u_norm, \
            MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE, \
            acc_max=acc_max) for rank in range(num_cpu)], start_method='spawn')
    
    #Create the evaluation environment
    if eval_environment == True:
        eps_schedule_eval = [eps_schedule[-1]]
        eval_env = DummyVecEnv([make_env(env_id=env_name, rank=0, seed=100, \
            obs_type=obs_type, \
            random_obs=random_obs, randomIC=randomIC, stochastic=stochastic, \
            termination=termination, NSTEPS=NSTEPS, NITER=niter_per_cpu/nminibatches, \
            eps_schedule=eps_schedule_eval, lambda_term=lambda_term, \
            Dvmax=Dvmax, tf=tf, omega=omega, \
            rKOZ=rKOZ, beta_cone=beta_cone, \
            r0=r0, v0=v0, \
            rTf=rTf, vTf=vTf, \
            dr0_max=dr0_max, dv0_max=dv0_max, \
            sigma_r=sigma_r, sigma_v=sigma_v, \
            sigma_u_rot=sigma_u_rot, sigma_u_norm=sigma_u_norm, \
            MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE, \
            acc_max=acc_max)])
        n_eval_episodes=100
        eval_freq=400.*n_steps
        if (stochastic == False and random_obs == False and randomIC == False):
            n_eval_episodes=1
            eval_freq=4.*n_steps
        eval_callback = EvalCallback(eval_env, n_eval_episodes=n_eval_episodes, \
                                best_model_save_path=out_folder, \
                                log_path=out_folder, eval_freq=eval_freq, \
                                deterministic=True)

    # Create the model
    if algorithm == "PPO": 
        if load_model == False:
            model = PPO2(policy=policy, env=env, 
                        n_steps=n_steps, nminibatches=nminibatches,
                        gamma=gamma, ent_coef=ent_coef, cliprange_vf=-1,
                        lam=lam, noptepochs=noptepochs,
                        learning_rate=learning_rate,
                        cliprange=clip_range,
                        tensorboard_log=tensorboard_log, verbose=1)
        else:
            model = PPO2.load(input_model, policy=policy, env=env, 
                        n_steps=n_steps, nminibatches=nminibatches,
                        gamma=gamma, ent_coef=ent_coef, cliprange_vf=-1,
                        lam=lam, noptepochs=noptepochs,
                        learning_rate=learning_rate,
                        cliprange=clip_range,
                        verbose=1)
            model.tensorboard_log = tensorboard_log
    elif algorithm == "A2C":
        model = A2C(policy=policy, env=env, n_steps=n_steps, 
                    gamma=gamma, ent_coef=ent_coef,
                    learning_rate=learning_rate_in,
                    lr_schedule=learning_rate_type,
                    tensorboard_log=tensorboard_log, verbose=1,)
    elif (platform.machine() != "ppc64le") and (algorithm == "DDPG"):
        model = DDPG(policy='MlpPolicy', env=env,
                    gamma=gamma, nb_train_steps=n_steps,
                    actor_lr=learning_rate_in, critic_lr=learning_rate_in, 
                    tensorboard_log=tensorboard_log, verbose=1)
    elif algorithm == "SAC":
        model = SAC(policy='MlpPolicy', env=env,
                    gamma=gamma, ent_coef=ent_coef, 
                    learning_rate=learning_rate,
                    batch_size=NSTEPS,
                    tensorboard_log=tensorboard_log, verbose=1)
    elif algorithm == "TD3":
        model = TD3(policy='MlpPolicy', env=env, 
                    gamma=gamma, batch_size=NSTEPS,
                    learning_rate=learning_rate,
                    tensorboard_log=tensorboard_log, verbose=1)
    elif (platform.machine() != "ppc64le") and (algorithm == "TRPO"):
        model = TRPO(policy=policy, env=env, 
                    gamma=gamma, timesteps_per_batch=NSTEPS,
                    lam=lam, entcoeff=ent_coef,
                    tensorboard_log=tensorboard_log, verbose=1)

    # Learning
    start_time = time.time()
    if eval_environment == True:
        model.learn(total_timesteps=niter, callback=eval_callback)
    else:
        model.learn(total_timesteps=niter)
    end_time = time.time()

    # Save solution
    model.save(trained_model_log)
    print("End Training.")

    # Save time
    f_out_time = open(out_folder + "time.txt", "w") # open file
    if eval_environment == True:
        f_out_time.write("%20s\t%20s\n" \
            % ("# elapsed time [s]", "best J"))
        f_out_time.write("%20.7f\t%20.7f\n" \
            % (end_time - start_time, eval_callback.best_mean_reward))
    else:
        f_out_time.write("%20s\n" \
            % ("# elapsed time [s]"))
        f_out_time.write("%20.7f\n" \
            % (end_time - start_time))
    f_out_time.close()

    # Post-process
    if postprocess == True:
        print("Post-processing\n")
        os.system('python main_rendezvous_load.py --folder ' + out_folder)
        if MonteCarlo == True and (stochastic == True or random_obs == True or randomIC == True):
            print("Monte Carlo\n")
            os.system('python main_rendezvous_MC.py --folder ' + out_folder)
