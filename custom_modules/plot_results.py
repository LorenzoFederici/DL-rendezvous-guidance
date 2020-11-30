import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, out_folder = None, title='Learning_Curve', save_plot = True):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param out_folder: (str) the location where the plot is saved (default: log_folder)
    :param title: (str) the title of the task to plot
    :param save_plot: (bool) save the plot as pdf?
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    matplotlib.rc('font', size=14)
    matplotlib.rc('text', usetex=True)
    #fig1 = plt.figure() #figsize=(10, 10))
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Cumulative reward')
    #plt.xscale('log')
    plt.yscale('symlog')
    plt.grid()
    #plt.title("Learning curve smoothed")
    if (save_plot):
        if (out_folder is None):
            plt.savefig(log_folder + title + ".pdf", dpi=300)
        else:
            plt.savefig(out_folder + title + ".pdf", dpi=300)