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

#Input settings file
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="sol_saved/sol_1/", \
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

#Problem data
Dvmax = float(Dvmax)                        #Maximum chaser Dv, km/s
omega = float(omega)                        #Mean motion of the target, rad/s
rKOZ = float(rKOZ)                          #radius of the keep out zone (KOZ), km
beta_cone = float(beta_cone)*np.pi/180.       #visibility cone semi-aperture, rad

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
rTf = np.array(rTf)
vTf = np.array(vTf)

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
model = PPO2.load(trained_model, policy=policy)

# Print graph and results
f_out = open(in_folder + "MCanalysis.txt", "w") # open file
f_out.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" % \
    ("# Dv [km/s]", "dx [km]", "dy [km]", "dz [km]", "dr [km]", \
    "dvx [km/s]", "dvy [km/s]", "dvz [km/s]", "dv [km/s]", "e_los_2 [km]", "e_los_inf [km]", "reward"))
f_out_traj_MC = open(in_folder + "Trajectory_MC.txt", "w") # open file
f_out_traj_MC.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# x [km]", "y [km]", "z [km]", "vx [km/s]", "vy [km/s]", "vz [km/s]"))
f_out_exp_traj = open(in_folder + "exp_trajectories.txt", "w") # open file
f_out_exp_traj.write("%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" \
        % ("# t [s]", "x [km]", "y [km]", "z [km]", \
        "vx [km/s]", "vy [km/s]", "vz [km/s]", \
        "Dvx [km/s]", "Dvy [km/s]", "Dvz [km/s]", "Dv [km/s]"))

#Nominal trajectory

#Reset environment
obs = env0.reset()
cumulative_reward = 0.
r_nom = []
v_nom = []
u_nom = []
u_nom_norm = []
e_los_nom = []
Dvtot = 0.
for i in range(NSTEPS):
    
    #Get current action
    action, _states = model.predict(obs, deterministic=True)

    #Get new observation
    obs, reward, done, info = env0.step(action)

    #Nominal spacecraft state, time and control
    r_nom.append(np.array([info["rx"], info["ry"], info["rz"]]))
    v_nom.append(np.array([info["vx"], info["vy"], info["vz"]]))
    u_nom.append(np.array([info["ux"], info["uy"], info["uz"]]))
    u_nom_norm.append(norm(u_nom[-1]))
    Dvtot += norm(u_nom[-1])

    #LOS constraint violation
    violated, max_viol = env0.operConstraints(r_nom[-1])
    e_los_nom.append(max_viol)

    #Update cumulative reward
    cumulative_reward += reward

#Final state
t = env0.t
r = env0.rk
v = env0.vkm

#Final LOS constraint violation
violated, max_viol = env0.operConstraints(r)
e_los_nom.append(max_viol)

rTf = np.array(rTf)
vTf = np.array(vTf)

#Final state, after DV
u_max = Dvmax
uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
rf = r
vf = v + uf
rf_nom = rf
vf_nom = vf
uf_nom = uf
Dvtot += norm(uf_nom)

#Final constraint violations
dr_vector = (rf - rTf)
dv_vector = (vf - vTf)
dr = norm(dr_vector)
dv = norm(dv_vector)

#LOS constraint violation
e_los_2 = norm(e_los_nom)
e_los_inf = max(e_los_nom)

f_out.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (Dvtot, dr_vector[0], dr_vector[1], dr_vector[2], dr, \
        dv_vector[0], dv_vector[1], dv_vector[2], dv, e_los_2, e_los_inf, cumulative_reward))

#MonteCarlo analysis
print("MC simulation")
Nsim = 1000
if stochastic == True:
    env0.stochastic = True
if random_obs == True:
    env0.random_obs = True
if randomIC == True:
    env0.randomIC = True
env0.seed(1000) #New seed

Dvtot_vec = []
dr_vec = []
dv_vec = []
rew_vec = []
e_los_2_vec = []
e_los_inf_vec = []
if (MTE == True) and (pr_MTE == 0):
    f_mte = open(in_folder + "MTE.txt", "w") # open file
    f_mte.write("%10s\t%10s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\n" % \
        ("# sim", "tk_mte", "dr_x", "dr_y", "dr_z", "dv_x", "dv_y", "dv_z"))
    tk_mte = []
    dr_mte = []
    dv_mte = []

beta_max = 0.
for sim in range(Nsim):
    
    print("Simulation number %d / %d" % (sim+1, Nsim))
    
    #Reset environment
    obs = env0.reset()
    cumulative_reward = 0.

    if MTE == True and pr_MTE == 0:
        new_MTE = False
    
    Dvtot = 0.

    e_los = []
    for i in range(NSTEPS):
        
        #Get current action
        action, _states = model.predict(obs, deterministic=True)

        #Get new observation
        obs, reward, done, info = env0.step(action)

        #Time and spacecraft state and control
        t = info["t"]
        r = np.array([info["rx"], info["ry"], info["rz"]])
        v = np.array([info["vx"], info["vy"], info["vz"]])
        u = np.array([info["ux"], info["uy"], info["uz"]])
        Dvtot += norm(u)

        #LOS constraint violation
        violated, max_viol = env0.operConstraints(r)
        e_los.append(max_viol)

        #Max. beta value
        beta_z_pos = abs(np.arctan(r[2] / r[1]))
        beta_z_neg = abs(np.arctan(-r[2] / r[1]))
        beta_x_pos = abs(np.arctan(r[0] / r[1]))
        beta_x_neg = abs(np.arctan(-r[0] / r[1]))
        beta_max_new = max([beta_z_pos, beta_z_neg, beta_x_pos, beta_x_neg])
        if beta_max_new > beta_max:
            beta_max = beta_max_new

        #Print expert trajectories
        f_out_exp_traj.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
            % (t, r[0], r[1], r[2], \
            v[0], v[1], v[2], \
            u[0], u[1], u[2], norm(u)))

        #Print trajectory segment
        N = 10
        dtj = dt / (N - 1)
        r_prop = r
        v_prop = v + u
        for j in range(N):
            f_out_traj_MC.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
                % (r_prop[0], r_prop[1], r_prop[2], \
                v_prop[0], v_prop[1], v_prop[2]))
            r_prop, v_prop = propagate_HCW(r0=r_prop, v0=v_prop, dt=dtj, omega=omega)
        
        if (MTE == True) and (pr_MTE == 0):
            tk = t/dt
            if (norm(u) == 0.) and (tk not in tk_mte):
                tk_mte.append(tk)
                new_MTE = True

        #Update cumulative reward
        cumulative_reward += reward

    #Final state
    t = env0.t
    r = env0.rk
    v = env0.vkm

    #Final LOS constraint violation
    violated, max_viol = env0.operConstraints(r)
    e_los.append(max_viol)

    #Final state, after DV
    u_max = Dvmax
    uf = min(norm(vTf - v), u_max)*(vTf - v)/norm(vTf - v)
    rf = r
    vf = v + uf
    Dvtot += norm(uf)

    #Final constraint violation
    dr_vector = (rf - rTf)
    dv_vector = (vf - vTf)
    dr = norm(dr_vector)
    dv = norm(dv_vector)
    
    dr_vec.append(dr)
    dv_vec.append(dv)
    Dvtot_vec.append(Dvtot)
    rew_vec.append(cumulative_reward)

    #LOS constraint violation
    e_los_2_vec.append(norm(e_los))
    e_los_inf_vec.append(max(e_los))

    if (MTE == True) and (pr_MTE == 0) and (new_MTE == True):
        dr_mte.append(dr)
        dv_mte.append(dv)
        dr_mte_vec = (rf - rTf)
        dv_mte_vec = (vf - vTf)
        f_mte.write("%10d\t%10d\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" % \
            (sim, np.round(tk_mte[-1]), dr_mte_vec[0], dr_mte_vec[1], dr_mte_vec[2], \
                dv_mte_vec[0], dv_mte_vec[1], dv_mte_vec[2]))
    
    f_out.write("%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" \
        % (Dvtot, dr_vector[0], dr_vector[1], dr_vector[2], dr, \
        dv_vector[0], dv_vector[1], dv_vector[2], dv, e_los_2_vec[-1], e_los_inf_vec[-1], cumulative_reward))
    f_out_traj_MC.write("\n\n");
    f_out_exp_traj.write("\n");

f_out.close()
f_out_traj_MC.close()
f_out_exp_traj.close()
if (MTE == True) and (pr_MTE == 0):
    f_mte.close()

if (MTE == True) and (pr_MTE == 0):
    sort_index = np.argsort(tk_mte)
    tk_mte.sort()
    dr_mte_sort = [dr_mte[i] for i in sort_index]
    dv_mte_sort = [dv_mte[i] for i in sort_index]
    dr_mte = dr_mte_sort
    dv_mte = dv_mte_sort

# DV tot
sorted_Dvtot = Dvtot_vec
sorted_Dvtot.sort()

# SR of final constraint and sigma-contours
SR_eps1 = 0
SR_eps2 = 0
SR_eps3 = 0
for j in range(Nsim):
    if (dr_vec[j] <= eps_schedule[-1]) and (dv_vec[j] <= eps_schedule[-1]):
        SR_eps1 += 1
        SR_eps2 += 1
        SR_eps3 += 1
    elif (dr_vec[j] <= 2.*eps_schedule[-1]) and (dv_vec[j] <= 2.*eps_schedule[-1]):
        SR_eps2 += 1
        SR_eps3 += 1
    elif (dr_vec[j] <= 10.*eps_schedule[-1]) and (dv_vec[j] <= 10.*eps_schedule[-1]):
        SR_eps3 += 1
max_err = [max(dr_vec[j], dv_vec[j]) for j in range(Nsim)]
max_err.sort()
one_sigma = round(68.27*Nsim/100)
two_sigma = round(95.45*Nsim/100)
three_sigma = round(99.73*Nsim/100)

# SR of LOS constraint (2-norm)
SR_los_2_eps0 = 0
SR_los_2_eps1 = 0
SR_los_2_eps2 = 0
SR_los_2_eps3 = 0
for j in range(Nsim):
    if (e_los_2_vec[j] <= 0.):
        SR_los_2_eps0 += 1
        SR_los_2_eps1 += 1
        SR_los_2_eps2 += 1
        SR_los_2_eps3 += 1
    elif (e_los_2_vec[j] <= eps_schedule[-1]):
        SR_los_2_eps1 += 1
        SR_los_2_eps2 += 1
        SR_los_2_eps3 += 1
    elif (e_los_2_vec[j] <= 2.*eps_schedule[-1]):
        SR_los_2_eps2 += 1
        SR_los_2_eps3 += 1
    elif (e_los_2_vec[j] <= 10.*eps_schedule[-1]):
        SR_los_2_eps3 += 1
sorted_e_los_2 = e_los_2_vec
sorted_e_los_2.sort()

# SR of LOS constraint (inf-norm)
SR_los_inf_eps0 = 0
SR_los_inf_eps1 = 0
SR_los_inf_eps2 = 0
SR_los_inf_eps3 = 0
for j in range(Nsim):
    if (e_los_inf_vec[j] <= 0.):
        SR_los_inf_eps0 += 1
        SR_los_inf_eps1 += 1
        SR_los_inf_eps2 += 1
        SR_los_inf_eps3 += 1
    elif (e_los_inf_vec[j] <= eps_schedule[-1]):
        SR_los_inf_eps1 += 1
        SR_los_inf_eps2 += 1
        SR_los_inf_eps3 += 1
    elif (e_los_inf_vec[j] <= 2.*eps_schedule[-1]):
        SR_los_inf_eps2 += 1
        SR_los_inf_eps3 += 1
    elif (e_los_inf_vec[j] <= 10.*eps_schedule[-1]):
        SR_los_inf_eps3 += 1
sorted_e_los_inf = e_los_inf_vec
sort_index = np.argsort(sorted_e_los_inf)
sorted_e_los_inf.sort()

#Print statistics
f_out_stats = open(in_folder + "MCstats.txt", "w") # open file
f_out_stats.write("%18s\t%12s\t%12s\t%12s\t%12s\n" % ("#", "Dv [km/s]", "dr [km]", "dv [km/s]", "reward"))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n" % ("mean", \
    np.mean(Dvtot_vec), np.mean(dr_vec), np.mean(dv_vec), np.mean(rew_vec)))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\t%12.7f\n\n" % ("sigma", \
    np.std(Dvtot_vec), np.std(dr_vec), np.std(dv_vec), np.std(rew_vec)))
f_out_stats.write("%21s\n" % ("# Total DV"))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\n" % ("#", "1-sigma", "2-sigma", "3-sigma"))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n\n" % ("eps_lvl[m/s]", \
    sorted_Dvtot[one_sigma-1]*1000., sorted_Dvtot[two_sigma-1]*1000., sorted_Dvtot[three_sigma-1]*1000.))
f_out_stats.write("%21s\n" % ("# Terminal constraints"))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\t%12s\t%12s\n" % ("#", "dr<=5e-4", "dv<=5e-4", "both<=5e-4", "both<=1e-3", "both<=5e-3"))
f_out_stats.write("%18s\t%12.2f\t%12.2f\t%12.2f\t%12.2f\t%12s\n" % ("SR[%]", \
    float(sum(map(lambda x : x<=eps_schedule[-1], dr_vec))/Nsim*100), \
    float(sum(map(lambda x : x<=eps_schedule[-1], dv_vec))/Nsim*100), float(SR_eps1/Nsim*100), \
    float(SR_eps2/Nsim*100), float(SR_eps3/Nsim*100)))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\n" % ("#", "1-sigma", "2-sigma", "3-sigma"))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n\n" % ("eps_lvl", \
    max_err[one_sigma-1], max_err[two_sigma-1], max_err[three_sigma-1]))
f_out_stats.write("%21s\n" % ("# LOS constraint"))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\t%12s\n" % ("#", "<=0", "<=5e-4", "<=1e-3", "<=5e-3"))
f_out_stats.write("%18s\t%12.2f\t%12.2f\t%12.2f\t%12.2f\n" % ("SR[%](2-norm)", \
    float(SR_los_2_eps0/Nsim*100), float(SR_los_2_eps1/Nsim*100), \
    float(SR_los_2_eps2/Nsim*100), float(SR_los_2_eps3/Nsim*100)))
f_out_stats.write("%18s\t%12.2f\t%12.2f\t%12.2f\t%12.2f\n" % ("SR[%](inf-norm)", \
    float(SR_los_inf_eps0/Nsim*100), float(SR_los_inf_eps1/Nsim*100), \
    float(SR_los_inf_eps2/Nsim*100), float(SR_los_inf_eps3/Nsim*100)))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\n" % ("#", "1-sigma", "2-sigma", "3-sigma"))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n" % ("eps_lvl(2-norm)", \
    sorted_e_los_2[one_sigma-1], sorted_e_los_2[two_sigma-1], sorted_e_los_2[three_sigma-1]))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n" % ("eps_lvl(inf-norm)", \
    sorted_e_los_inf[one_sigma-1], sorted_e_los_inf[two_sigma-1], sorted_e_los_inf[three_sigma-1]))
f_out_stats.write("%18s\t%12s\t%12s\t%12s\n" % ("#", "min [km]", "max [km]", "mean [km]"))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n" % ("err_los(2-norm)", \
    min(e_los_2_vec), max(e_los_2_vec), np.mean(e_los_2_vec)))
f_out_stats.write("%18s\t%12.7f\t%12.7f\t%12.7f\n\n" % ("err_los(inf-norm)", \
     min(e_los_inf_vec), max(e_los_inf_vec), np.mean(e_los_inf_vec)))
f_out_stats.write("%18s\t%12.7f\n\n" % ("max.beta[deg]", beta_max*180./np.pi))
f_out_stats.close()

if graphs:
    #Trajectory figure
    os.system("gnuplot -e \"indir='" + str(args.folder) + "'\" \"PlotFiles/plot_traj_MC.plt\"")
    os.system("latexmk -pdf")
    os.system("latexmk -c")
    os.system("rm *.eps *.tex *-inc-eps-converted-to.pdf")
    os.system("mv *.pdf " + args.folder)


print("Results printed, graphs plotted.")