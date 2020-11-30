import os
import platform
import warnings
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import generate_expert_traj
if platform.machine() == "ppc64le":
    from stable_baselines import PPO2, A2C, SAC, TD3
else:
    from stable_baselines import PPO2, A2C, DDPG, SAC, TD3, TRPO
from custom_modules.custom_policies import CustomPolicy_2x32, CustomPolicy_3x64, CustomPolicy_4x128, \
    CustomLSTMPolicy, CustomPolicy_2x81, CustomPolicy_3_var
from custom_modules.learning_schedules import linear_schedule
from custom_modules.plot_results import plot_results
from custom_modules.set_axes_equal_3d import set_axes_equal
import argparse
import gym_rendezvous
from gym_rendezvous.envs.pyHCW import propagate_HCW

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# Input data
graphs = True
expert_traj = False

#Input settings file
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="sol_saved/-1traj_8nb_500epochs_95frac_1/", \
    help='Input model folder')
args = parser.parse_args()
settings_file = args.folder + "settings.txt"

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

#Model parameters
policy = globals()[policy]

#Input Model and Output folders
in_folder = "./" + args.folder
if os.path.isfile(in_folder + "best_model.zip"):
    logname = "best_model"
else:
    logname = "final_model"
trained_model = in_folder + logname
plot_folder = in_folder

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

#Epsilon constraint schedule
eps_schedule = [eps_schedule[-1]]

# Environment creation
env0 = gym.make(id=env_name, \
            obs_type=obs_type, \
            random_obs=False, randomIC=False, stochastic=False, \
            termination=termination, NSTEPS=NSTEPS, NITER=NSTEPS, \
            eps_schedule=eps_schedule, lambda_term=lambda_term, \
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

# Load model
npzfile = np.load(trained_model + ".zip")
model = PPO2.load(trained_model, policy=policy)
if expert_traj:
    print("\nGenerating expert trajectories:\n")
    generate_expert_traj(model, in_folder + 'expert_rendezvous', env=env0, n_episodes=100)
    #npzfile = np.load(in_folder + 'expert_rendezvous.npz')
    
# Print graph and results
f_out = open(in_folder + "Simulation.txt", "w") # open file
f_out.write("Environment simulation\n\n") 
f_out_traj = open(in_folder + "Trajectory.txt", "w") # open file
f_out_u = open(in_folder + "control.txt", "w") # open file
f_out_traj.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# x", "y", "z", "vx", "vy", "vz"))
f_out_u.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
    % ("# t", "x", "y", "z", "Dvx", "Dvy", "Dvz", "Dvnorm", "Dvmax"))
np.set_printoptions(precision=3)

#Unzip and save evaluations file
evalutation_file_npz = in_folder + "evaluations.npz"
if os.path.isfile(evalutation_file_npz):
    npzfile = np.load(evalutation_file_npz)
    f_out_eval = open(in_folder + "evaluations.txt", "w") # open file
    f_out_eval.write('%20s\t%20s\t%20s\t%20s\n' \
        % ("# training step", "mean reward", "std reward", "mean ep. length"))
    for i in range(len(npzfile['timesteps'])):
        f_out_eval.write('%20d\t%20.5f\t%20.5f\t%20d\n' \
        % (npzfile['timesteps'][i], np.mean(npzfile['results'][i]), np.std(npzfile['results'][i]), \
            np.mean(npzfile['ep_lengths'][i])))
    f_out_eval.close()

    if graphs:
        matplotlib.rc('font', size=12)
        matplotlib.rc('text', usetex=True)
        fig0 = plt.figure()
        plt.plot(npzfile['timesteps'], np.mean(npzfile['results'], axis=1),'o-')
        plt.xlabel('Training step number')
        plt.ylabel('Mean reward')
        plt.yscale('symlog')
        plt.grid()
        plt.savefig(in_folder + "Evaluations.pdf", dpi=300)

if (graphs):
    matplotlib.rc('font', size=18)
    matplotlib.rc('text', usetex=True)
    fig1 = plt.figure()
    ax1 = fig1.gca()

# Environment simulation

#Reset environment
obs = env0.reset()
cumulative_reward = 0.
u_tot = 0.
beta_max = 0.
for i in range(NSTEPS):
    
    #Get current action
    action, _states = model.predict(obs, deterministic=True)

    #Get new observation
    obs, reward, done, info = env0.step(action)

    #Spacecraft state, time and control
    r = np.array([info["rx"], info["ry"], info["rz"]])
    v = np.array([info["vx"], info["vy"], info["vz"]])
    t = info["t"]
    u = np.array([info["ux"], info["uy"], info["uz"]])
    u_max = Dvmax
    u_tot += norm(u)

    #Print trajectory information
    f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
    f_out.write("norm(r) = " + str(norm(r)) + "\n")
    f_out.write("norm(v) = " + str(norm(v)) + "\n")
    f_out.write("norm(u/u_max) = " + str(norm(u)/u_max) + "\n")
    f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

    #Max. beta value
    beta_z_pos = abs(np.arctan(r[2] / r[1]))
    beta_z_neg = abs(np.arctan(-r[2] / r[1]))
    beta_x_pos = abs(np.arctan(r[0] / r[1]))
    beta_x_neg = abs(np.arctan(-r[0] / r[1]))
    beta_max_new = max([beta_z_pos, beta_z_neg, beta_x_pos, beta_x_neg])
    if beta_max_new > beta_max:
        beta_max = beta_max_new

    #Print trajectory segment
    N = 10
    dtj = dt / (N - 1)
    r_prop = r
    v_prop = v + u
    for j in range(N):
        f_out_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
            % (r_prop[0], r_prop[1], r_prop[2], \
            v_prop[0], v_prop[1], v_prop[2]))
        r_prop, v_prop = propagate_HCW(r0=r_prop, v0=v_prop, dt=dtj, omega=omega)

    #Print control
    f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (t, r[0], r[1], r[2], u[0], u[1], u[2], norm(u), u_max))

    if graphs:
        #Plot control
        ax1.stem([t/tf], [norm(u)*1000.], '-k')

    #Update cumulative reward
    cumulative_reward += reward

#Final state
t = env0.t
r = env0.rk
v = env0.vkm

rTf = np.array(rTf)
vTf = np.array(vTf)

#Final state, after DV
uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
u_tot += norm(uf)
rf = r
vf = v + uf

#Print final state
f_out.write("t_step = " + str(np.round(t/dt)) + "\n")
f_out.write("norm(r) = " + str(norm(r)) + "\n")
f_out.write("norm(v) = " + str(norm(v)) + "\n")
f_out.write("norm(u/u_max) = " + str(norm(uf)/u_max) + "\n")
f_out.write("cum_reward = " + str(cumulative_reward) + "\n\n")

#Print final control
f_out_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (rf[0], rf[1], rf[2], vf[0], vf[1], vf[2]))
f_out_u.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (tf, rf[0], rf[1], rf[2], uf[0], uf[1], uf[2], norm(uf), u_max))

if graphs:
    #Plot final control
    ax1.stem([tf/tf], [norm(uf)*1000.], '-k')

f_out.write("Final position: r = " + str(rf) + " km\n")
f_out.write("Final velocity: v = " + str(vf) + " km/s\n")
f_out.write("Final target position: rT = " + str(rTf) + " km\n")
f_out.write("Final target velocity: vT = " + str(vTf) + " km/s\n")
f_out.write("Final position error: dr = " + str(norm(rf - rTf)) + " km\n")
f_out.write("Final velocity error: dv = " + str(norm(vf - vTf)) + " km/s\n")
f_out.write("Max LOS angle: beta_max = " + str(beta_max*180./np.pi) + " deg\n")
f_out.write("Total Dv: Dv = " + str(u_tot) + " km/s\n")
f_out.write("Final time: t = " + str(t) + " s\n\n")

f_out.close()
f_out_traj.close()
f_out_u.close()

if graphs:
    #Control figure
    plt.xlabel('$t/t_f$')
    plt.ylabel('$\\Delta v$, [m/s]')
    plt.grid()
    fig1.savefig(plot_folder + "control.pdf", dpi=300, bbox_inches='tight')

    #Trajectory figure
    os.system("gnuplot -e \"indir='" + str(args.folder) + "'\" \"PlotFiles/plot_traj.plt\"")
    os.system("latexmk -pdf")
    os.system("latexmk -c")
    os.system("rm *.eps *.tex *-inc-eps-converted-to.pdf")
    os.system("mv *.pdf " + args.folder)


print("Results printed, graphs plotted.")