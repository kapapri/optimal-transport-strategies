# ==================================================
# Training loop
# ==================================================
# Training of the neural network and calculation of the resultant current

import numpy as np
import math
import random
import torch
from tqdm import tqdm
# For DQN_Check and Simple_Environment:
#from simple_environment import SimpleLattice
from environment import LatticeTASEP
from DQN import optimization

# Reproducibility
# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)

class Training():
    def __init__(self, env_params, rl_params, device, log):
        self.env_params = env_params
        self.rl_params = rl_params
        self.device = device
        self.log = log
        self.env = LatticeTASEP(env_params)
        # For DQN_Check and Simple_Environment:
        #self.env = SimpleLattice(env_params)
        self.steps_done = 0

    def select_action(self, state, policy_net):
     # greedy epsilon policy implementation    
        EPS_START = self.rl_params['EPS_START']
        EPS_END = self.rl_params['EPS_END']
        EPS_DECAY = self.rl_params['EPS_DECAY']

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold: # Explotation: best known action
            with torch.no_grad():      
                # - t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward. We sum one because the particles are labelled starting from 1
                action = policy_net(state).max(1)[1].view(1,1) + 1 # view (1,1) makes a tensor of dim 1 with one element: tensor([[value]])
                
                # For DQN_Check and Simple_Environment:
                #action = policy_net(state).max(1)[1].view(1,1)               
                return action
            
        else:  # Exploration: random action
            return torch.tensor([[np.random.randint(1, self.env_params['N'])]], device=self.device, dtype=torch.float32)


    def select_particle_random(self):
     # random policy implementation (no learning)
        return torch.tensor([[np.random.randint(1, self.env_params['N'])]], device=self.device, dtype=torch.float32) 

    def simulate(self, runsNumber, episodes, memory, policy_net, target_net, optimizer):
        Lx = self.env_params['Lx']
        Ly = self.env_params['Ly']
        N = self.env_params['N']
        size = Lx * Ly
     # Memory allocation
        CurrentAlongConsecutive = np.zeros(episodes*runsNumber, dtype=np.float32)  # Current during all runs
        CurrentAlongPerRun = np.zeros((runsNumber, episodes), dtype=np.float32)  # Array of current during one run

        # Learning tracking
        lossTot = np.zeros(runsNumber, dtype=np.float32)
        rewardsTot = np.zeros(runsNumber, dtype=np.float32)
        # Animations
        JumpRate_movie = np.zeros((episodes*N + 1, Ly, Lx), dtype=np.float32) # Frames of each movement attempt in one episode
        JumpRate_short_movie = np.zeros((episodes + 1, Ly, Lx), dtype=np.float32)  # Frames of set of movements after one episode
        JumpRate_short_Tot = np.zeros((runsNumber, episodes + 1, Ly, Lx), dtype=np.float32)  # Array of episodes frames for each run
        i = 0    
        for iwalk in tqdm(range(runsNumber)):
            if self.log == True:
                print('- Run Number', iwalk)
         # Memory allocation
            CurrentAlong = np.zeros(episodes, dtype=np.float32)
            # Learning tracking
            loss_count = 0
            reward_count = 0        
         # Initialize the environment and get its state    
            Init_System, velocities, NN_input, info = self.env.reset()
            state = torch.tensor(np.reshape(NN_input, (1, 2*Lx*Ly)), dtype=torch.float32, device=self.device)
            # Animation storage
            JumpRate_movie[0] = self.env.get_jumping_rates()
            JumpRate_short_movie[0] = self.env.get_jumping_rates()
            k = 1            
            if self.log == True:
                print('Initial System:\n', Init_System)
                print('    Initial NN_input: \n', NN_input) 

            for istep in range(episodes):
                if self.log == True:
                    print(' - Episode Number', istep)
                Along_count = 0
                for moveAttempt in range(N): # To make a move over all particles
                    if self.log == True:
                        print('  - moveAttempt', moveAttempt)                               
                    #action =  Training.select_particle_random(self) # random action
                    action = Training.select_action(self, state, policy_net) # greedy epsilon policy
                    if self.log == True:
                        print('    particle chosen:', int(action))                    
                    System, velocities, observation, reward, terminated, truncated, info = self.env.step(action.item(), log = self.log)
                    reward = torch.tensor([reward], device=self.device, dtype=torch.float64)
                 # Counters
                    reward_count += reward # Cumulated reward
                    Along_count += info["Along_Steps"]
                    #print('Along_count', Along_count)
                    next_state = torch.tensor(np.reshape(observation, (1, 2*Lx*Ly)), dtype=torch.float32, device=self.device)
                 # Store the transition in memory
                    memory.push(state, action, next_state, reward)

                    if self.log == True:
                        print('    next state: \n', System)
                        print('    final reward:', float(reward))                                
                        print('    reward_count:', float(reward_count))
                        print('    NN_input: \n', observation) 
                        #print('Replay Memory Random Sample', memory.sample(1))

                # Move to the next state
                    state = next_state
                # Optimization of the NNs
                    loss_count = optimization(memory, policy_net, target_net, self.device, self.rl_params['BATCH_SIZE'], self.rl_params['GAMMA'], self.rl_params['TAU'], optimizer, loss_count)
                # Storing results
                    # Frames for the movie at each moveAttempt
                    JumpRate_movie[k] = self.env.get_jumping_rates()
                    k += 1        
                    # Frames for the movie at the last episode
                    JumpRate_short_movie[istep + 1] = self.env.get_jumping_rates()
                    
                    # Computes currents
                    CurrentAlong[istep] = Along_count / size  
                    # CurrentAlong[istep] = Along_count / Lx  # It is divided by the size and not the number of particles because 
                                                                #if you have only one particle in the system, the current through some part of the system, 
                                                                #say through X=0 periodic boundary would be very close to zero for large L
                                                                #but if you guys divide the number of updates by number of particles in the system, 
                                                                #then the current would be always high, even for small number of particles
                CurrentAlongConsecutive[i] = Along_count / size
                i += 1                                                                   
            rewardsTot[iwalk] = reward_count # Cumulative reward over one episode
            lossTot[iwalk] = loss_count / (N * episodes) # Average loss in each episode
            JumpRate_short_Tot[iwalk] = JumpRate_short_movie
            CurrentAlongPerRun[iwalk] = CurrentAlong

        print('Complete')
        return CurrentAlongConsecutive, CurrentAlongPerRun, lossTot, rewardsTot, JumpRate_movie, JumpRate_short_movie, JumpRate_short_Tot