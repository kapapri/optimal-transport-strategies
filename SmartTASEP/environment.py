# ==================================================
# Environment
# ==================================================
# Environment: Grid with discrete sites that can have value 0 (empty) or a number corresponding to the particle label.
#  It can start with a random distribution of n agents or a chess fashion filling homogeneously the lattice.
# Optimization: Maximizing the current along the X-axis 
# Action space: Each action consists in choosing a specific particle and applying a TASEP process to it. Each episode lasts a constant number of movement attempts (max_steps).

import numpy as np
import math
import random
from numba import njit
import gymnasium as gym
from gymnasium import spaces
import pygame

# ==================================================
# Generation of the System and Utilities
# ==================================================
def checkerboard(Lx, Ly, n = False): 
    """ 
    Filling the lattice with particles alternatively (chess fashion).
    
    It returns two arrays of size (Lx*Ly):
    - Map: it describes where the particles are with zeros (empty site) and ones (occupied site) in the lattice.
    - System: it describes where the particles are with the number of particle assigned
    - JumpRateGrid: Same structure as System but instead of ones where there is a particle, it has the corresponding jumping rates sampled from a truncated gaussian.
    """
    n = Lx * Ly // 2 # number of particles
    Labels = [x + 1 for x in range(n)]  # Particle labels from 1 to n. Zero is an unoccupied site 
    
    Map = np.zeros((Ly, Lx), dtype=np.int32)  # Ly vectors with Lx (zero) components 
    Map[::2, ::2] = 1  # Set even rows and even columns to 1
    Map[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    
    System = np.zeros((Ly, Lx), dtype=np.int32)  # Ly vectors with Lx (zero) components 
    JumpRateGrid = np.zeros((Ly,Lx), dtype=np.float32)

    k = 0
    for i in range(Ly):
        for j in range(Lx):
            if Map[i][j] == 1:
                JumpRateGrid[i][j] = random.choice([0.1, 1])
                System[i][j] = Labels[k]
                k += 1


    return System, JumpRateGrid


@njit
def random_system(Lx, Ly, mu, sigma, n): #Still to update
    """
    Filling the lattice (Lx*Ly) randomly with n particles
    
    It returns two arrays of size (Lx*Ly):
    - System: it describes where the particles are with zeros (empty site) and ones (occupied site) in the lattice.
    - JumpRateGrid: Same structure as System but instead of ones where there is a particle, it has the corresponding jumping rates sampled from a truncated gaussian.
    """
    assert n <= (Lx*Ly) # We cannot have more particles than avaliable sites
    Map = np.concatenate((np.ones(n, dtype=np.float32), np.zeros(Lx * Ly - n, dtype=np.float32))) 
    np.random.shuffle(Map)
    Map = Map.reshape((Ly, Lx))
    
    Labels = [x + 1 for x in range(n)]  # Particle labels from 1 to n. Zero is an unoccupied site 

    System = np.zeros((Ly, Lx), dtype=np.int32)  # Ly vectors with Lx (zero) components     
    JumpRateGrid = np.zeros((Ly,Lx), dtype=np.float32)
    k = 0        
    for i in range(Ly):
        for j in range(Lx):
            if Map[i][j] == 1:
                JumpRateGrid[i][j] = trunc_gaussian_sample(mu, sigma) # probability of jumping forward or transversally
                System[i][j] = Labels[k]                
                k += 1

    return System, JumpRateGrid

def locating_particle(Lx, Ly, action, System):
    # Search the coordinates of that particle given by the action
    for i in range(Ly):
        for j in range(Lx):
            if (action) == System[i][j]:
                X = i
                Y = j         
    return X, Y

class LatticeTASEP(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, env_params):
        self.window_size = 512  # The size of the PyGame window
                
        # Choosing initial conditions
        if env_params['mode'] == "chess":
            mode = checkerboard
        elif env_params['mode'] == "random":
            mode = random_system
            
        self.mode = mode # Initial distribution of the map
        self.Lx = env_params['Lx']  # The horizontal length
        self.Ly = env_params['Ly']  # The vertical length
        self.n = env_params['N'] # number of elements
        self.mu = env_params['mu']
        self.sigma = env_params['fixed_sigma']
        
        self.max_steps = env_params ['max_steps'] # number of movements in each iteration        
        self.step_counter = 0        
        self.reward = 0
        
        self.Along_count = 0
        self.Transv_count = 0        
        self.CurrentAlong = 0
        self.CurrentTransv = 0
 
        # Lx*Ly grid with 0 and 1 values corresponding to empty or occupied site
        self.observation_space = spaces.MultiDiscrete(np.ones([self.Ly, self.Lx])*(self.n + 1)) # it has Lx*Ly dimensions and a range value of 0 to n      
        # e.g: array([[ 1, 0, 2, 0],
        #             [ 0, 3, 0, 4],
        #             [ 5, 0, 6, 0]])
    
        # We have n actions, corresponding to choosing a specific particle to move
        self.action_space = spaces.Discrete(self.n, start = 1)  

        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = env_params['render_mode']

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # Translates the environment’s state into an observation
    def _get_obs(self):
        return self.System
    
    def get_jumping_rates(self):
        return self.JumpRateGrid   

    # Auxiliary information
    def _get_info(self):
        return {}
    
    def get_size(self):
        return {self.Lx,self.Ly}    


    # ==================================================
    # Reset and Step Methods
    # ==================================================        

    def reset(self, seed=None, options=None):
        
        self.truncated = False
        self.reward = 0
        self.step_counter = 0
        
        self.Along_count = 0
        self.Transv_count = 0          
        
        # To seed self.np_random
        super().reset(seed=seed)
 
        # Choose the agents location with the mode function
        self.System, self.JumpRateGrid= self.mode(self.Lx, self.Ly, self.n)

        # NN input
        FastChannel = np.zeros((self.Ly, self.Lx), dtype=np.int32)  
        SlowChannel = np.zeros((self.Ly, self.Lx), dtype=np.int32)             
        for i in range(self.Ly):
            for j in range(self.Lx):
                if self.JumpRateGrid[i][j] == 1:
                    FastChannel[i][j] = 1
                elif self.JumpRateGrid[i][j] != 0:
                    SlowChannel[i][j] = 1
        NN_input = np.concatenate((FastChannel, SlowChannel))


        observation= self._get_obs()
        velocities = self.get_jumping_rates()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, velocities, NN_input, info
    

    def step(self, action, log = False):       
        
        """
        It updates the arrays System and JumpRateGrid when the particle at [X, Y] lattice jumps once following TASEP 
        and counts the jump done after depending on the direction it is realized, along the main current (x-axis) or perpendicular (y-axis)

        """
        Along_step = 0
        Transv_step  = 0
        
        Lx = self.Lx
        Ly = self.Ly
        self.step_counter += 1
        System = self.System.copy() # Apparently without that the initial observation updates every time a step is done
        JumpRateGrid = self.JumpRateGrid.copy()
        reward = float(self.reward)
        
        # Search the coordinates of that particle (action)
        X,Y = locating_particle(Lx, Ly, action, System)
        
        """ 
        The lattice is labelled from the left top to the right 
         as: action = 0, 1, 2    or    [X,Y]= (0,0), (0,1), (0,2) 
                      3, 4, 5                 (1,0), (1,1), (1,2)
        """           
        
        # TASEP, deciding direction: p=1/2 of jumping forward, p=1/4 of jumping up or down                  
        dice = random.randint(0, 1)
        if dice == 0:
            JumpAlong = 0
            JumpTransverse = 2 # No jump up or down
        else:
            JumpAlong = 2 # No jump forward
            JumpTransverse = random.randint(0, 1)                        

        # Getting velocity/jumping_rate of the particle at [X][Y]
        JumpRate = JumpRateGrid[X][Y] 
        dice_jump = random.random() # Deciding if it moves or not

        # Simple implementation of Periodic boundary conditions
        xPrev = Ly - 1 if X == 0 else X - 1        
        xNext = 0 if X == Ly - 1 else X + 1
        yNext = 0 if Y == Lx - 1 else Y + 1
        yPrev = Lx - 1 if Y == 0 else Y - 1

        
     # BORRAMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE EN ALGUN MOMENTO
     # Modifications:
        #JumpRate = 1 # It always jumps

        # 1D TASEP        
        # JumpAlong = 0 # Jumps only forwards
        # JumpTransverse = 2 # No jump up or down                
        
        # Updating the system
        if dice_jump < JumpRate:
            if JumpAlong == 0: # tries to hop forward
                if System[X][yNext] == 0:  # if next site is free
                    System[X][Y], System[X][yNext] = System[X][yNext], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[X][yNext] = JumpRateGrid[X][yNext], JumpRateGrid[X][Y]
                    Along_step = 1
                    reward = 1
                    if log == True:
                        print("    Hop forward")
                        print("    instant reward", reward)                         
                    
                else:                  
                    reward = -10
                    if log == True:
                        print("    Site occupied: No ha podido saltar palante :(")
                        print("    instant reward", reward)                         

                    
            if JumpTransverse == 0: # hop up
                if System[xPrev][Y] == 0: 
                    System[X][Y], System[xPrev][Y] = System[xPrev][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xPrev][Y] = JumpRateGrid[xPrev][Y], JumpRateGrid[X][Y]
                    Transv_step = 1
                    reward = 0
                    if log == True:                    
                        print("    Hop up")
                        print("    instant reward", reward)                   

                else:
                    reward = -10
                    if log == True:                    
                        print("    Site occupied: Attempt to hop up")
                        print("    instant reward", reward)                         

            if JumpTransverse == 1: # hop down
                if System[xNext][Y] == 0: 
                    System[X][Y], System[xNext][Y] = System[xNext][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xNext][Y] = JumpRateGrid[xNext][Y], JumpRateGrid[X][Y]
                    Transv_step = -1
                    reward = 0
                    if log == True:                                        
                        print("    Hop down")
                        print("    instant reward", reward)                   

                    
                else:
                    reward = -10
                    if log == True:                                        
                        print("    Site occupied: Attempt to hop down")
                        print("    instant reward", reward)  
                                             
            # Clustering reward
            # Update of boundary conditions !!!!!!!!!!!!!!!!!!!
            xPrev = Ly - 1 if xPrev == 0 else xPrev - 1        
            xNext = 0 if xNext == Ly - 1 else xNext + 1
            yNext = 0 if yNext == Lx - 1 else yNext + 1
            yPrev = Lx - 1 if yPrev == 0 else yPrev - 1

            d_c = 1 # Equilibrium distance
            K = 1 # LJ constant
            r = 1/d_c # Nearest neighbours

            Dfront,Dback,Dup,Ddown = 0, 0, 0, 0
            LJfront,LJback,LJup,LJdown = 0, 0, 0, 0
            fast_count = 0
            slow_count = 0


            # Nearest neighbours 
            print(System)          
            if System[X][yNext] != 0:
                print('forward occupied')
                print(System[X][yNext])
                Dfront = abs(JumpRateGrid[X][yNext] - JumpRate)
                #LJfront = 1/r**2 - 1/r

                if JumpRateGrid[X][yNext] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yNext] != 0:
                    slow_count += 1 

            if System[X][yPrev] != 0:
                print('back occupied')
                print(System[X][yPrev])                
                Dback = abs(JumpRateGrid[X][yPrev] - JumpRate)
                #LJback = 1/r**2 - 1/r

                if JumpRateGrid[X][yPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yPrev] != 0:
                    slow_count += 1

            if System[xNext][Y] != 0:
                print('down occupied')
                print(System[xNext][Y])                
                Dup = abs(JumpRateGrid[xNext][Y] - JumpRate)
                #LJup = 1/r**2 - 1/r

                if JumpRateGrid[xNext][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xNext][Y] != 0:
                    slow_count += 1                   

            if System[xPrev][Y] != 0:
                print('up occupied')
                print(System[xPrev][Y])                

                Ddown = abs(JumpRateGrid[xPrev][Y] - JumpRate)   
                #LJdown = 1/r**2 - 1/r

                if JumpRateGrid[xPrev][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xPrev][Y]  != 0:
                    slow_count += 1               
                    
            JumpRate_diff = (-1)*float(Dfront + Dback + Dup + Ddown)
            #LJ_function = (-K)*int(LJfront + LJback + LJup + LJdown)
            LJ_function = 0
            reward += JumpRate_diff + LJ_function            

            if log == True:
                print("fast_count", fast_count)
                print("slow_count", slow_count)

                # print("Dfront", Dfront)
                # print("Dback", Dback) 
                # print("Dup", Dup) 
                # print("Ddown", Ddown) 

                print("    JumpRate_diff", JumpRate_diff)         
                #print("LJ_function", LJ_function)                   
                print("    instant reward after jr difference", reward)


            if JumpRate == 1:
                reward += fast_count - 10*slow_count

            if JumpRate != 0:
                reward += -10*fast_count + slow_count
                                
            if log == True:
                print("    instant reward after counting numbers", reward)   

        else:
            if log == True:
                print("    Sin velocidad") 

                                    
                    
        # NN input
        FastChannel = np.zeros((Ly, Lx), dtype=np.int32)  
        SlowChannel = np.zeros((Ly, Lx), dtype=np.int32)             
        for i in range(Ly):
            for j in range(Lx):
                if JumpRateGrid[i][j] == 1:
                    FastChannel[i][j] = 1
                elif JumpRateGrid[i][j] != 0:
                    SlowChannel[i][j] = 1
        NN_input = np.concatenate((FastChannel, SlowChannel))

        self.JumpRateGrid = JumpRateGrid
        self.System = System
        self.reward = reward
            
        truncated = False
        if self.step_counter >= self.max_steps: # To stop the step after (max_steps) iterations
            truncated = True
                
        observation = self._get_obs()
        state = self.get_jumping_rates()        
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        info["Along_Steps"] = Along_step
        info["Transv_Steps"] = Transv_step
                

        return observation, state, NN_input, self.reward, False, truncated, info
    

    # ==================================================
    # Rendering
    # ==================================================        

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
        if self.render_mode == "ansi":
            return self._render_text()
        
    def _render_text(self):
        
        return np.array2string(self.board)

    #-------------------------------------------------------------------------------------------------------------------------------------------------------    

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()