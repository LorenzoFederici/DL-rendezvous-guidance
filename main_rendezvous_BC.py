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
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2, A2C, DDPG, SAC, TD3, TRPO
from stable_baselines.common.callbacks import EvalCallback
from custom_modules.custom_policies import CustomPolicy_2x32, CustomPolicy_3x64, CustomPolicy_4x128, \
    CustomLSTMPolicy, CustomPolicy_2x81, CustomPolicy_3_var
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

    #Input settings file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_BC', type=str, default="settingsBC1.txt", \
        help='Input BC settings file')
    parser.add_argument('--settings_RL', type=str, default="settingsRL1.txt", \
        help='Input RL settings file')    
    args = parser.parse_args()
    settings_file = "./settings_files/" + args.settings_BC
    settingsRL_file = "./settings_files/" + args.settings_RL

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

    #Behavior Cloning parameters
    RL = bool(int(RL_training)) #the BC pre-trained network is then trained by RL?
    n_batches_BC = int(n_batches_BC) #the number of mini-batches for behavior cloning
    batch_size = int(NSTEPS*n_batches_BC) #batch size for behavior cloning
    traj_limitation = int(traj_limit) #the number of trajectories to use (if -1, load all)
    train_fraction = float(train_frac) #the train validation split (0 to 1)
    n_epochs = int(n_epochs) #number of iterations on the training set
    learning_rate = float(lr) #learning rate
    adam_epsilon = float(adam_eps) #the epsilon value for the adam optimizer

    #Output folders and log files
    n_sol = 1
    if RL:
        out_folder_root = "./sol_saved/BC+RL/"
    else:
        out_folder_root = "./sol_saved/"
    out_folder_root += expert_name + "/"
    out_folder_root += str(traj_limitation) + "traj_"  + \
                str(n_batches_BC) + "nb_" + \
                str(n_epochs) + "epochs_" + \
                str(int(train_fraction * 100)) + "frac_"
    out_folder = out_folder_root + str(n_sol) + "/"
    while os.path.exists(out_folder):
        n_sol += 1
        out_folder = out_folder_root + str(n_sol) + "/"
    
    os.makedirs(out_folder, exist_ok=True)
    trained_model_name = "final_model"
    trained_model_log = out_folder + trained_model_name
    os.system("cp " + settings_file + " " + out_folder)
    os.system("mv " + out_folder + args.settings_BC + " " + out_folder + "settings.txt")

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
    dt = tf/NSTEPS               #s, time-step

    #Reference initial state
    r0 = [rx_nom[0], ry_nom[0], rz_nom[0]] #km, initial relative chaser position
    v0 = [vx_nom[0], vy_nom[0], vz_nom[0]] #km/s, initial relative chaser velocity

    #Target state
    rTf = [rx_nom[-1], ry_nom[-1], rz_nom[-1]]      #km, final target position
    vTf = [vx_nom[-1], vy_nom[-1], vz_nom[-1]]      #km/s, final target velocity

    # Create the environment
    env0 = gym.make(id=env_name, \
                obs_type=obs_type, \
                random_obs=random_obs, randomIC=randomIC, stochastic=stochastic, \
                termination=termination, NSTEPS=NSTEPS, NITER=NSTEPS, \
                eps_schedule=[eps_schedule[-1]], lambda_term=lambda_term, \
                Dvmax=Dvmax, tf=tf, omega=omega, \
                rKOZ=rKOZ, beta_cone=beta_cone, \
                r0=r0, v0=v0, \
                rTf=rTf, vTf=vTf, \
                dr0_max=dr0_max, dv0_max=dv0_max, \
                sigma_r=sigma_r, sigma_v=sigma_v, \
                sigma_u_rot=sigma_u_rot, sigma_u_norm=sigma_u_norm, \
                MTE=MTE, pr_MTE=pr_MTE, max_MTE=max_MTE, \
                acc_max=acc_max)
    env0.seed(0)

    # Create expert trajectories
    expert_folder = "./expert_traj/"
    expert_file = expert_folder + expert_name + ".npz" #File with expert trajectories
 
    # Create expert dataset
    dataset = ExpertDataset(expert_path=expert_file,
                        traj_limitation=traj_limitation, batch_size=batch_size,
                        train_fraction=train_fraction)

    model = PPO2(policy=policy, env=env0, verbose=1)
    
    # Pretrain the PPO2 model
    start_time = time.time()
    model.pretrain(dataset, n_epochs=n_epochs, 
                learning_rate=learning_rate, adam_epsilon=adam_epsilon)
    end_time = time.time()

    # Save solution
    model.save(trained_model_log)
    print("End Training.")

    # Save time
    f_out_time = open(out_folder + "time.txt", "w") # open file
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

    # Training with RL
    if RL == True:
        print("Training by RL\n")
        os.system('python main_rendezvous_RL.py --settings ' + settingsRL_file + " --input_model_folder " + out_folder)
    