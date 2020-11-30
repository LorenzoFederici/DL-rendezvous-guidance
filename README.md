# Deep Learning Techniques for Autonomous Spacecraft Guidance during Proximity Operations

This code implements a Deep Learning (DL) framework to deal with the autonomous guidance of a chaser spacecraft during a minimum-fuel, time-fixed, impulsive linear rendezvous maneuver,
with fixed initial and final relative conditions.
Specifically, the aim of the mission is to come in close proximity of a passive target body, starting from given initial conditions.
Moreover, two type of operational constraints are considered: a spherical keep-out zone, 
representing the space occupied by the target, and a visibility cone
the chaser must move through during the final part of the maneuver, 
which guarantees that the chaser spacecraft is always visible from the target.
The chaser trajectory is divided into a number of segments, and 
the Hill-Clohessy-Wiltshire (HCW) equations are used to propagate the chaser relative
motion between any two impulsive Dvs. 

<img align="right" src="https://github.com/LorenzoFederici/DL-rendezvous-guidance/blob/main/images/traj_MC.png" width="500" />

The environment can be deterministic or stochastic (i.e., with state,
observation and/or control uncertainties, as well as with
stochastic dynamical perturbations).
The initial state of the chaser can be fixed or uniformly scattered around a reference value.
The resulting (stochastic) optimal control problem is solved by using a Deep Neural Network (DNN) trained by either Behavioral Cloning (BC) or Reinforcement Learning (RL), or both.

The program is based on [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), an open-source library containing a set of improved implementations of RL algorithms based on [OpenAI Baselines](https://github.com/openai/baselines).
All necessary information can be found in the corresponding GitHub repository.

Moreover, the program uses [OpenMPI](https://www.open-mpi.org/) message passing interface implementation to run different environment realizations in parallel on several CPU cores during training.

## Installation

To correctly set up the code on Linux (Ubuntu), please follow the present instructions:

1. First, you need Python3 and the system packages CMake, OpenMPI and zlib. You can install all these packages with:
    ```
    $ sudo apt-get update
    $ sudo apt-get install cmake openmpi-bin libopenmpi-dev python3-dev zlib1g-dev
    ```
2. After installing [Anaconda](https://www.anaconda.com/distribution/), you can create a virtual environment with a specific Python version (3.6.10) by using conda:
    ```
    $ conda create --name myenv python=3.6.10
    ```
    where `myenv` is the name of the environment, and activate the environment with:
    ```
    $ conda activate myenv
    ```
    When the conda environment is active, your shell prompt is prefixed with (myenv).
    The environment can be deactivated with:
    ```
    $ conda deactivate
    ```

3. Install all required packages in the virtual environment by using pip and the requirement file in the current repository:

    ```
    (myenv)$ cd /path-to-cloned-GitHub-repository/DL-rendezvous-guidance/
    (myenv)$ pip install -r requirements.txt
    ```
4. Install the gym RL environment with pip:
    ```
    (myenv)$ pip install -e gym-rendezvous
    ```
Now, you can verify that all packages were successfully compiled and installed by running the following command:

```
(myenv)$ python main_rendezvous_validate.py
```
If this command executes without any error, then your DL-rendezvous-guidance installation is ready for use.

The plots are realized by using gnuplot. For this reason, you may need to install gnuplot on your machine:
```
$ sudo apt-get install -y gnuplot
```
Eventually, a LaTeX distribution should be installed on the computer in order to correctly visualize all graphs and plots. As an example, to install [Tex Live LaTeX](https://www.tug.org/texlive/) distribution on Ubuntu use the following command.
```
$ sudo apt-get install texlive-full
```

## User Guide

The program is composed by 4 main Python scripts (`main_rendezvous_(...).py`), the RL environment (in `gym-rendezvous/`), created by using the [OpenAI Gym](https://gym.openai.com/) toolkit, and a number of additional Python modules (in `custom_modules/`) with useful functions.

Specifically:

1. `main_rendezvous_RL.py` is the main file that must be called to train the DNN by RL.

    All the environment settings, the RL hyper-parameter values and the DNN configuration must be specified in an external text file with ad-hoc formatting, 
    to be placed in folder `settings_files/` and given as input to the script.
    For example, to start training the agent by RL with the settings specified in file `settings_files/settingsRL.txt`, you can use the following command:
    ```
    (myenv)$ python main_rendezvous_RL.py --settings "settingsRL.txt"
    ```
    The information that must be included in the settings file, together with the right formatting, are described in detail in `settings_files/README.md`. 
    A sample settings file, named `settingsRL.txt`, is already included in folder `settings_files/` for convenience; you can look at it to better 
    understand how to prepare new "legal" settings files from scratch. 

    The specific linear rendezvous mission to study must be specified in a .dat file to be placed in folder `missions/`.
    The file should contain at least three rows, the first containing the columns headers, and the other two specifying the time (`t`), the relative chaser position (`x`, `y`, `z`) and velocity (`xdot`, `ydot`, `zdot`) in a RTN frame centered on the target body at the beginning and at the end of the mission, respectively. All quantities are assumed to be in s, km and km/s.
    The name of the mission file must be specified in the settings file.
    For example, file `MSR.dat` contains the initial and final chaser state for a typical Mars Sample Return mission.
    
    It is also possible to re-train by RL, with the same or different settings, a pre-trained model. In this case,
    the program must be called with the command:
    ```
    (myenv)$ python main_rendezvous_RL.py --settings "settings-name.txt" --input_model_folder "relative-path-to-input-model-folder/"
    ```
    where `settings-name.txt` is the name of the settings file which contains the new training settings, and `reletive-path-to-input-model-folder/` is the relative path (i.e., starting from the cloned repo directory) of the folder which contains the pre-trained model, named `best_model.zip` or `final_model.zip`

    At the end of the training, the final RL-trained model (`final_model.zip`), the best RL-trained model found according to the evaluation callback (`best_model.zip`), together with all other output files, are saved in directory `sol_saved/(i)env_(j)nb_(k)Msteps_(l)/`, where numbers (i), (j) and (k) depends on training settings specified in the settings file, and number (l) depends on how many solutions of the same kind are already contained in `sol_saved/` folder.
    In presence of a pre-trained model, the outputs of training are instead saved in directory `path-to-input-model-folder/RL/(i)env_(j)nb_(k)Msteps_(l)/`.
    The output RL folder that is obtained by launching the script with settings file `settings_files/settingsRL.txt` can be already found in directory `sol_saved/`.

2. `main_rendezvous_BC.py` is the main file that must be called to train the DNN by BC.
    
    All the environment settings, the BC hyper-parameter values and the DNN configuration must be specified in an external text file with ad-hoc formatting, 
    to be placed in folder `settings_files/` and given as input to the script.
    For example, to start training the DNN by BC with the settings specified in file `settings_files/settingsBC.txt`, use the following command:
    ```
    (myenv)$ python main_rendezvous_BC.py --settings_BC "settingsBC.txt"
    ```
    The information that must be included in the settings file, together with the right formatting, are described in detail in `settings_files/README.md`. 
    A sample settings file, named `settingsBC.txt`, is already included in folder `settings_files/` for convenience; you can look at it to better 
    understand how to prepare new "legal" settings files from scratch.
    The training is performed by using the expert trajectories in the `.npz` file contained in the `expert_traj/` folder, whose name is specified in the settings file. Two different training sets (`cone20-S1.npz`, `cone20-eyelash-S1.npz`) for the environment specified in file `settings_files/settingsBC.txt` are already included in the `expert_traj/` directory.
    Similar expert trajectories sets for the problem of interest can be directly created by the user by following [this guide](https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html) by Stable Baselines.

    It is also possible to pre-train a DNN by BC on a given environment, and, subsequently, re-train the same network with RL on the same (or a different) environment. In this case,
    the program must be called with the command:
    ```
    (myenv)$ python main_rendezvous_BC.py --settings_BC "settings-BC-name.txt" --settings_RL "settings-RL-name.txt"
    ```
    where `settings-BC-name.txt` and `settings-RL-name.txt` are the names of the settings file which contains the BC and RL training settings, respectively.

    At the end of the training, the final BC-trained model (`final_model.zip`), the best BC-trained model found according to the evaluation callback (`best_model.zip`), together with all other output files, are saved in directory `sol_saved/(i)traj_(j)nb_(k)epochs_(h)frac_(l)/`, where numbers (i), (j), (k) and (h) depends on training settings specified in the settings file, and number (l) depends on how many solutions of the same kind are already contained in `sol_saved/` folder.
    If the re-training by RL is also performed, the outputs of training are instead saved in directory `sol_saved/BC+RL/(i)traj_(j)nb_(k)epochs_(h)frac_(l)/`.
    The output BC folder that is obtained by launching the script with settings file `settings_files/settingsBC.txt` can be already found in directory `sol_saved/`.

3. `main_rendezvous_load.py` is the file that allows generating the output files and the plots referred to the nominal (robust) trajectory, obtained by
running a trained policy in a deterministic version of the environment.
For doing so, it is sufficient to run the command:
    ```
    (myenv)$ python main_rendezvous_load.py --folder "relative-path-to-trained-model-folder/"
    ```
    All output files and graphs are saved in the same directory.

4. `main_rendezvous_MC.py` is the file that allows performing a Monte Carlo simulation of a given policy in a stochastic environment.
For doing so, it is sufficient to run the command:
    ```
    (myenv)$ python main_rendezvous_MC.py --folder "relative-path-to-trained-model-folder/"
    ```
    The graphs and output files are saved in the same directory.
    The settings of the stochastic environment where the MC simulations are performed are taken from the `settings.txt` file already included in the input directory.
    
Both `main_rendezvous_load.py` and `main_rendezvous_MC.py` are, by default, called at the end of the training procedure triggered by both script `main_rendezvous_RL.py` and `main_rendezvous_BC.py`.

Enjoy the code!
