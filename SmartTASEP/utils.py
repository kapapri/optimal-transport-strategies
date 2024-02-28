import numpy as np
from numba import njit

import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==================================================
# Saving functions
# ==================================================
def save_data_to_file(file_path, dicti):
    """
    Save the data storage dictionary to a file using pickle.

    Parameters:
    - file_path: The path to the file where the data will be saved.
    """
    if not os.path.exists(os.path.dirname("trainings/")):
        os.makedirs(os.path.dirname("trainings/"))
            
    # if not os.path.exists(f"trainings/{file_path}"):
    #     with open(f"trainings/{file_path}", "bw") as file:
    #         file.write("{}")
    else:
        with open(f"trainings/{file_path}", 'bw') as file:
            pickle.dump(dicti, file)

def load_data_from_file(file_path, dicti):
    """
    Load the data storage dictionary from a file using pickle.

    Parameters:
    - file_path: The path to the file from which the data will be loaded.
    """
    with open(f"trainings/{file_path}", 'rb') as file:
        dicti = pickle.load(file)
        return dicti
    
class Storage():
    def __init__(self, env_params, rl_params, runsNumber, episodes):
        Lx = env_params['Lx']
        Ly = env_params['Ly']
        fixed_sigma = env_params['fixed_sigma']
        BATCH_SIZE = rl_params['BATCH_SIZE']
        GAMMA = rl_params['GAMMA']
        EPS_START = rl_params['EPS_START']
        EPS_END = rl_params['EPS_END']
        EPS_DECAY = rl_params['EPS_DECAY']
        TAU = rl_params['TAU']
        LR = rl_params['LR']

        self.hyperparameters = {'size':[Lx, Ly], 'sigma': fixed_sigma, 'runsNumber':runsNumber, 'episodes': episodes, 
                             'BATCH_SIZE': BATCH_SIZE , 'GAMMA': GAMMA, 'EPS_START':EPS_START, 'EPS_END':EPS_END, 'EPS_DECAY':EPS_DECAY, 'TAU':TAU, 'Learning Rate':LR}
        self.training_storage = {}
        self.data_storage = {}

    def update_dictionaries(self, runsNumber, episodes, policy_net, target_net, memory, CurrentAlongTot, CurrentAlongPerRun, lossTot, rewardsTot, JumpRate_short_Tot, Load, data_storage):

        self.training_storage['policy_NN'] = policy_net.state_dict()
        self.training_storage['target_NN'] = target_net.state_dict()
        self.training_storage['memory'] = memory.whole_list()
        self.training_storage['hyperparameters'] = self.hyperparameters
        self.training_storage['hyperparameters']['runsNumber'] = runsNumber
        self.training_storage['hyperparameters']['episodes'] = episodes

        if Load == True:
            # print(data_storage['consecutive_current'])
            self.data_storage['consecutive_current'] = np.concatenate((data_storage['consecutive_current'], CurrentAlongTot))
            self.data_storage['CurrentAlongPerRun'] = np.concatenate((data_storage['CurrentAlongPerRun'], CurrentAlongPerRun))
            self.data_storage['average_loss'] = np.concatenate((data_storage['average_loss'], lossTot))
            self.data_storage['cumulative_reward'] = np.concatenate((data_storage['cumulative_reward'], rewardsTot))
            self.data_storage['JumpRate_short_Tot'] = np.concatenate((data_storage['JumpRate_short_Tot'], JumpRate_short_Tot))

        else:
            self.data_storage['consecutive_current'] = CurrentAlongTot
            self.data_storage['CurrentAlongPerRun'] = CurrentAlongPerRun
            self.data_storage['average_loss'] = lossTot
            self.data_storage['cumulative_reward'] = rewardsTot
            self.data_storage['JumpRate_short_Tot'] = JumpRate_short_Tot

        # self.data_storage['consecutive_current'] = CurrentAlongTot        
        # self.data_storage['average_loss'] = lossTot
        # self.data_storage['cumulative_reward'] = rewardsTot
        self.data_storage['hyperparameters'] = self.hyperparameters
        self.data_storage['hyperparameters']['runsNumber'] = runsNumber
        self.data_storage['hyperparameters']['episodes'] = episodes

        return self.training_storage, self.data_storage


# ==================================================
# Visualization functions
# ==================================================
class Visualization():
    def __init__(self, data_storage):
        data_storage = load_data_from_file('data_storage.pkl', data_storage)
        self.CurrentAlongTot = data_storage['consecutive_current']
        self.CurrentAlongPerRun = data_storage['CurrentAlongPerRun']
        self.lossTot = data_storage['average_loss']
        self.rewardsTot = data_storage['cumulative_reward']
        self.runsNumber = data_storage['hyperparameters']['runsNumber']
        self.episodes = data_storage['hyperparameters']['episodes']
        #For plot with several sigmas
        # env = LatticeTASEP(env_params)
        # playground = Dynamics(env_params, rl_params, device, log = False)
        # memory = ReplayMemory(10000)        

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def create_combined_plot(self, Lx, Ly, sigma, LR, lines, save = False):      
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

        # Plot for current
        x1 = np.array(range(self.CurrentAlongTot.size))
        ax1.plot(x1[5:], self.CurrentAlongTot[5:], '.', label='Parallel Current')

        # Choose the window size for the moving average
        window_size = 25
        # Calculate the moving average
        moving_avg = self.moving_average(self.CurrentAlongTot[5:], window_size)
        # Adjust time to match the moving average length
        adjusted_time = x1[:len(moving_avg)] 
        
        ax1.plot(adjusted_time, moving_avg)    
        ax1.set_title('Current per unit time (N move attempts)')
        ax1.set_xlabel('episodes*Runs')    
        #ax1.legend()

        # # Plot for current overlapping
        # x1 = np.array(range(self.CurrentAlongPerRun[0].size))
        # orig_map = plt.cm.get_cmap('hsv') 
        # reversed_map = orig_map.reversed() 
        # colors = reversed_map(np.linspace(0, 1, self.runsNumber))
        # if self.runsNumber - lines < 0:
        #     start = 0
        # else:
        #     start = self.runsNumber - lines
        # for i in range(start, self.runsNumber):
        #     # ax1.plot(x1, self.CurrentAlongPerRun[i,:], '.')

        #     # Choose the window size for the moving average
        #     window_size = 10
        #     # Calculate the moving average
        #     moving_avg = self.moving_average(self.CurrentAlongPerRun[i,:], window_size)
        #     # Adjust time to match the moving average length
        #     adjusted_time = x1[:len(moving_avg)]
        #     ax1.plot(adjusted_time, moving_avg, linewidth=2, label='Run %i' % (i), color=colors[i])
                
        #     ax1.set_title('Current per unit time (N move attempts)')
        #     ax1.set_xlabel('Episodes (episodes) [t=1/N]')    
        #     ax1.legend(ncol=2)        

        # Plot for loss
        x2 = np.array(range(self.lossTot.size))
        ax2.plot(x2, self.lossTot, label='Loss', color = 'firebrick')
        ax2.set_title('Average loss over each episode')
        ax2.set_xlabel('Runs')

        # Plot for reward
        x3 = np.array(range(self.rewardsTot.size))    
        ax3.plot(x3, self.rewardsTot, label='Rewards', color = 'forestgreen')
        ax3.set_title('Cumulated reward over each episode')
        ax3.set_xlabel('Runs')
        # ax3.set_ylim([-1000, 300])

        plt.tight_layout()  # Adjust layout to prevent overlap
        if save == True:
            plt.savefig(f'Pictures/System{Lx}x{Ly}_Runs{self.runsNumber}_episodes{self.episodes}_LearningRate{LR}.pdf')
        
        plt.show()

    def loss_plot(self, loss):
        plt.cla()
        plt.xlabel('Time')
        plt.ylabel(f'Loss')
    
        x_axis=np.array(range(self.runsNumber * self.episodes * self.N))
        plt.plot(x_axis[5:], loss[5:], label='Loss')

    def create_animation(self, Frames_movie):
        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        
        cv0 = Frames_movie[0]
        im = ax.imshow(cv0, cmap="gnuplot")
        cb = fig.colorbar(im, cax=cax)
        
        tx2 = ax.set_title('Frame 0 after one episode of the last run', y=1)
        
        ax.axis('off')
        plt.close()  # To not have the plot of frame 0

        def animate(frame):
            arr = Frames_movie[frame]
            vmax = 1
            vmin = np.min(arr)
            im.set_data(arr)
            im.set_clim(vmin, vmax)
            cb.ax.set_ylabel('Jumping Rate')
            tx2.set_text('Frame {0}'.format(frame))

        ani = FuncAnimation(fig, animate, frames=len(Frames_movie), repeat=False)
        return ani

    # ==================================================
    # Benchmark plots (with the sigma fixed)
    # ==================================================
    def current_plot(self):
        plt.cla()
        plt.title(f"TASEP. Gaussian jumping rate over {self.runsNumber} runs")
        plt.xlabel('Time')
        plt.ylabel(f'Current Forward')
        plt.grid(True)
        
        #The steady current of particle J, through a bond i, i+1 is given by the rate r multiplied by the probability that there is a particle at site i, and site i+1 is vacant
        r = 0.5 #jumping rate
        p = 0.5 #probability forward
        # prob_occ= 1 # we make a transition only when the site chosen is occupied
        prob_vac= 1/2 #density of particles in the system
        prob_occ= 1/2 # we make a transition only when the site chosen is occupied    
        f = np.array([r*p*prob_occ*prob_vac] * self.episodes)
        
        x_axis=np.array(range(self.episodes))
        
        # Choose the window size for the moving average
        window_size = 50
        # Calculate the moving average
        moving_avg = self.moving_average(self.CurrentAlongTot[5:], window_size)
        # Adjust time to match the moving average length
        adjusted_time = x_axis[:len(moving_avg)]

        
        plt.plot(x_axis, self.CurrentAlongTot, '.', label='Parallel Current')
        #plt.plot(x_axis[5:], CurrentTransvTot[5:], '.', label='Perpendicular Current')
        #plt.plot(x_axis, f, label='Theoretical Parallel Current', color = 'red')
        plt.plot(adjusted_time, moving_avg)
        plt.legend()
        
        # plt.savefig(f'Pictures/{init}_sigma={sigma}-Currents_{Ly}x{Lx}.pdf')

    # ==================================================
    # Benchmark plots (with several sigmas)
    # ==================================================  
    def currents_sigmas(self, sigmas): 
        Currents = np.zeros((self.episodes, len(sigmas)), dtype= np.float32)
        i = 0
        for sigma in sigmas:
            CurrentAlongTot, lossTot, rewardsTot, JumpRate_movie, JumpRate_short_movie = playground.simulate(self.runsNumber, self.episodes, memory, policy_net, target_net, optimizer)

            Currents[:,i] = self.CurrentAlongTot
            i += 1
        return Currents
        
    # Current comparison
    def currents_sigmas_plot(self, N): #To correct
        """
        It returns a plot with currents simulated from different sigmas
        
        """
        plt.cla()
        plt.title(f"TASEP. Gaussian jumping rate over {self.runsNumber} runs")
        plt.xlabel('Time')
        plt.ylabel(f'Current. {N} moves averaged')
        plt.grid(True)
        
        #sigmas = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5.0]
        sigmas = [0.0001, 0.1, 0.6, 5.0]
        Currents = self.currents_sigmas(sigmas)
        times = np.array(range(self.totalMCS))
        for i in range(len(sigmas)):        
            plt.plot(times, Currents[:, i], label=f"$\\sigma = {sigmas[i]}$")
            
        plt.legend()
        #plt.savefig(f'Pictures/Current_Different_Sigmas/Currents_{Ly}x{Lx}.pdf') 
    def average_run_current(self, sigmas):
        currents = np.zeros(len(sigmas), dtype = np.float32)    
        i = 0
        for sigma in sigmas:
            currents[i] = np.mean(self.CurrentAlongTot)
            i += 1
            #sleep(0.1) # to avoid #IOStream.flush timed out
            print("Percentage", (int(i/len(sigmas)*100)))
        return currents

    def average_run_current_over_sigmas(self, N):
        plt.cla()
        plt.xlabel('Standard Deviation, $\\sigma$ (log scale)')
        plt.xscale('log')
        plt.ylabel(f'Current. {N} moves averaged')
        plt.title(f"TASEP. Gaussian jumping rate over {self.runsNumber} runs (40x20)")
        plt.grid(True)
        
        sigmas = np.logspace(-2, 1, 60, dtype=np.float32)        
        currents = self.average_run_current(sigmas)
        plt.plot(sigmas, currents)
        
        #np.save(f"Data/Current_Different_Sigmas/AverageCurrentVsSigma_{Ly}x{Lx}.npy", currents)
        #np.save(f"Data/Current_Different_Sigmas/AverageCurrentVsSigma_sigmas{Ly}x{Lx}.npy", sigmas)
        #plt.savefig(f"Pictures/Current_Different_Sigmas/AverageCurrentVsSigma_{Ly}x{Lx}.pdf") 