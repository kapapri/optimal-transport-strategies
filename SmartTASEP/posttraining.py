# ==================================================
# Posttraining
# ==================================================
# Calculation of the resultant current using montecarlo with a trained neural network

import numpy as np
import math
import random
import torch
from tqdm import tqdm
from environment import LatticeTASEP
from DQN import optimization

# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)

class Posttraining():
    def __init__(self, env_params, device, log):
        self.env_params = env_params
        self.device = device
        self.log = log
        self.env = LatticeTASEP(env_params)
        self.cumulative_prob = 0
        self.action_counter = 0

    def best_action(self, state, policy_net):
        with torch.no_grad():      
            # - t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward. We sum one because the particles are labelled starting from 1
            action = policy_net(state).max(1)[1].view(1,1) + 1 # view (1,1) makes a tensor of dim 1 with one element: tensor([[value]])           
        return action.item()

    def best_action_wo_repeating(self, state, policy_net):
        with torch.no_grad():      
            Q_vector = policy_net(state)
            probabilities = (Q_vector.numpy() / np.sum(Q_vector.numpy()))[0] # Transforming the Q-value into probabilities
            chosen_index = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]  # k=1 ensures that only one action is chosen.
            action = chosen_index + 1
        return action

    def select_particle_random(self):
     # random policy implementation (no learning)
        return torch.tensor([[np.random.randint(1, self.env_params['N'])]], device=self.device, dtype=torch.float32)

    def simulate(self, runsNumber, totalMCS, policy_net, policy):
        Lx = self.env_params['Lx']
        Ly = self.env_params['Ly']
        N = self.env_params['N']
        size = Lx * Ly

     # Memory allocation
        CurrentAlongTot = np.zeros(totalMCS, dtype=np.float32) 
        CurrentTransvTot = np.zeros(totalMCS, dtype=np.float32)
        JumpRate_movie = np.zeros((totalMCS*N + 1, Ly, Lx), dtype=np.float32) # Frames of each movement attempt in one MCS
        RunMovie = np.zeros((runsNumber, totalMCS*N + 1, Ly, Lx), dtype=np.float32)  # Each run has totalMCS frames

        JumpRate_short_movie = np.zeros((totalMCS + 1, Ly, Lx), dtype=np.float32)  # Frames of set of movements after one MCS
        RunMovie_short = np.zeros((runsNumber, totalMCS + 1, Ly, Lx), dtype=np.float32)  # Each run has totalMCS frames
        
        for iwalk in tqdm(range(runsNumber)):
            if self.log == True:
                print('- Run Number', iwalk)
         # Memory allocation
            CurrentAlong = np.zeros(totalMCS, dtype=np.float32)
            CurrentTransv = np.zeros(totalMCS, dtype=np.float32)
            # Initialize the environment and get its state    
            Init_System, velocities, NN_input, info = self.env.reset()
            state = torch.tensor(np.reshape(NN_input, (1, 2*Lx*Ly)), dtype=torch.float32, device=self.device)
            # Animation storage
            JumpRate_movie[0] = self.env.get_jumping_rates()
            JumpRate_short_movie[0] = self.env.get_jumping_rates()
            if self.log == True:
                print('Initial System:\n', Init_System)
                print('    Initial NN_input: \n', NN_input) 
            
            k = 1            
            for istep in range(totalMCS):
                if self.log == True:
                    print(' - MCS Number', istep)
                Along_count = 0
                Transv_count = 0

                for moveAttempt in range(N): # To make a move over all particles
                    if self.log == True:
                        print('  - moveAttempt', moveAttempt)                               
                    
                    if policy == "random":
                        action = Posttraining.select_particle_random(self)
                    if policy == "bestWoRepeating":
                        action = Posttraining.best_action_wo_repeating(self, state, policy_net)
                    if self.log == True:
                        print('    particle chosen:', int(action))

                    System, velocities, observation, reward, terminated, truncated, info = self.env.step(action, log = self.log)
                 # Updates
                    Along_count += info["Along_Steps"]
                    Transv_count += info["Transv_Steps"]
                    #print('Along_count', Along_count)
                    next_state = torch.tensor(np.reshape(observation, (1, 2*Lx*Ly)), dtype=torch.float32, device=self.device)

                    if self.log == True:
                        print('    next state: \n', System)
                        print('    NN_input: \n', observation) 

                    # Move to the next state
                    state = next_state
                    # Frames for the movie at each move attempt
                    JumpRate_movie[k] = self.env.get_jumping_rates()
                    k += 1                        
                    # Frames for the movie after N movements
                    JumpRate_short_movie[istep + 1] = self.env.get_jumping_rates()
                    # Computes currents
                    CurrentAlong[istep] = Along_count / size  
                    CurrentTransv[istep] = Transv_count / size  
           
            RunMovie[iwalk] = JumpRate_movie
            RunMovie_short[iwalk] = JumpRate_short_movie
            #print('CurrentAlong', CurrentAlong)
            for dt in range(totalMCS):
                CurrentAlongTot[dt] += CurrentAlong[dt]
                CurrentTransvTot[dt] += CurrentTransv[dt]

            #print('CurrentAlongTot Before', CurrentAlongTot)

        # Simulation results output
        for dt in range(totalMCS):
            CurrentAlongTot[dt] /= runsNumber
            CurrentTransvTot[dt] /= runsNumber
        #print('CurrentAlongTot After', CurrentAlongTot)

        print('Complete')
        return CurrentAlongTot, CurrentTransvTot, RunMovie_short, RunMovie