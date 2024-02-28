# ==================================================
# Environment
# ==================================================
# Observational space: Grid with discrete sites that can have value 0 (empty) or a number corresponding to the particle label.
#                      It is initiallized with half density of particles randomly distributed or in a chess fashion.
# Action space: Each action consists in choosing a specific particle and applying a TASEP process to it.
# Optimization: Maximizing the current along the X-axis

import numpy as np
import math
import random
from numba import njit

# Reproducibility
# random.seed(1234)
# np.random.seed(1234)

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
    n = Lx * Ly // 2 # Number of particles
    Labels = [x + 1 for x in range(n)]  # Particle labels from 1 to n. Zero is an unoccupied site 
    
    Map = np.zeros((Ly, Lx), dtype=np.int32)  # Ly vectors with Lx (zero) components 
    Map[::2, ::2] = 1  # Set even rows and even columns to 1
    Map[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    
    System = np.zeros((Ly, Lx), dtype=np.int32)  
    JumpRateGrid = np.zeros((Ly,Lx), dtype=np.float32)
    Values = np.concatenate((np.ones(n // 2), np.full(n - (n // 2), 0.5)))
    k = 0
    for i in range(Ly):
        for j in range(Lx):
            if Map[i][j] == 1:
                
                JumpRateGrid[i][j] = 1 # Only fast particles
                # JumpRateGrid[i][j] = Values[k] # Half fast, half slow
                # JumpRateGrid[i][j] = random.choice([0.5, 1]) # Two types randomly distributed
                
                System[i][j] = Labels[k]
                k += 1

    return System, JumpRateGrid


@njit
def random_system(Lx, Ly, n):
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
    Values = np.concatenate((np.ones(n // 2), np.full(n - (n // 2), 0)))    
    k = 0        
    for i in range(Ly):
        for j in range(Lx):
            if Map[i][j] == 1:
                JumpRateGrid[i][j] = 1 # Only fast particles
                # JumpRateGrid[i][j] = Values[k] # Half fast, half slow
                # JumpRateGrid[i][j] = random.choice([0.5, 1]) # Two types randomly distributed
                
                System[i][j] = Labels[k]                
                k += 1

    return System, JumpRateGrid

def locating_particle(Lx, Ly, action, System): # Search the coordinates of that particle given by the action
    for i in range(Ly):
        for j in range(Lx):
            if (action) == System[i][j]:
                X = i
                Y = j         
    return X, Y

class LatticeTASEP():
    def __init__(self, env_params):
        
        # Choosing initial conditions
        if env_params['mode'] == "chess":
            mode = checkerboard
        elif env_params['mode'] == "random":
            mode = random_system
            
        self.mode = mode
        self.Lx = env_params['Lx']  # The horizontal length
        self.Ly = env_params['Ly']  # The vertical length
        self.n = env_params['N'] # number of particles
        
        self.max_steps = env_params ['max_steps'] # number of movements in each iteration, set to the number of particles     
        self.step_counter = 0        
        
        self.Along_count = 0
        self.Transv_count = 0        
        self.CurrentAlong = 0
        self.CurrentTransv = 0
        self.reward = 0

    # Translates the environment’s state into an observation
    def _get_obs(self):
        return self.System
    
    def get_jumping_rates(self):
        return self.JumpRateGrid   

    # Current data
    def _get_info(self):
        return {}
    
    def get_size(self):
        return {self.Lx, self.Ly}    


    # ==================================================
    # Reset and Step Methods
    # ==================================================        
    def reset(self, seed=None, options=None):
        
        self.truncated = False
        self.step_counter = 0
        
        self.Along_count = 0
        self.Transv_count = 0          
        self.reward = 0
 
        # Choose the agents location with the mode function
        self.System, self.JumpRateGrid = self.mode(self.Lx, self.Ly, self.n)

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

        observation = self._get_obs()
        velocities = self.get_jumping_rates()
        info = self._get_info()

        return observation, velocities, NN_input, info
    

    def step(self, action, log = False):       
        
        """
        It updates the arrays System and JumpRateGrid when the particle at [X, Y] lattice jumps once following TASEP 
        and counts the jump done after depending on the direction it moves, along the main current (x-axis) or perpendicular (y-axis)
        """
        Along_step = 0
        Transv_step  = 0
        exclusion_counter = 0
        
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
        dice = random.randint(0, 1) # Dice with 0 or 1 value
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
        
     # Modifications:
        #JumpRate = 1 # It always jumps
       # 1D TASEP:
        # JumpAlong = 0 # Jumps only forwards
        # JumpTransverse = 2 # No jump up or down                
        
        # Updating the system
        if dice_jump < JumpRate:
            if JumpAlong == 0: # tries to hop forward
                if System[X][yNext] == 0:   # forward site free
                    System[X][Y], System[X][yNext] = System[X][yNext], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[X][yNext] = JumpRateGrid[X][yNext], JumpRateGrid[X][Y]
                    Along_step = 1
                    
                    # Rewards from checking surroundings
                    if System[xPrev][Y] == 0 and System[xNext][Y] == 0: # up, right and down free
                        reward = 4
                    elif System[xNext][Y] != 0 and System[xPrev][Y] != 0: # Only forward free
                        reward = 3
                    elif System[xNext][Y] != 0 and System[xPrev][Y] == 0: # Only up and forward free
                        reward = 3
                    else: # Only down and forward free
                        reward = 3
                        
                    # Update of coordinates 
                    Y = yNext

                    if log == True:
                        print("    Hop forward")
                        print("    instant reward", reward)                         
                    
                else: # forward site occupied           
                    if System[xPrev][Y] == 0 and System[xNext][Y] == 0: # up and down free
                        reward = -2
                    elif System[xNext][Y] != 0 and System[xPrev][Y] != 0: # All surrounded
                        reward = -2
                    elif System[xNext][Y] != 0 and System[xPrev][Y] == 0: # Only up free
                        reward = -2
                    else:
                        reward = -2                                      # Only down free                    
                    
                    if log == True:
                        print("    Site occupied: it couldn't jump forward :(")
                        print("    instant reward", reward)                         

                    
            if JumpTransverse == 0: # hop up
                if System[xPrev][Y] == 0: # top site free
                    System[X][Y], System[xPrev][Y] = System[xPrev][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xPrev][Y] = JumpRateGrid[xPrev][Y], JumpRateGrid[X][Y]
                    Transv_step = 1
                    
                    # Rewards from checking surroundings
                    if System[xNext][Y] == 0 and System[X][yNext] == 0: # All free
                        reward = 0
                    elif System[xNext][Y] != 0 and System[X][yNext] != 0: # Only up free
                        reward = 0
                    elif System[xNext][Y] != 0 and System[X][yNext] == 0: # Up and forward free
                        reward = 0
                    else:
                        reward = 0
                        
                    # Update of coordinates 
                    X = xPrev

                    if log == True:                    
                        print("    Hop up")
                        print("    instant reward", reward)                   

                else: # top site occupied
                    if System[xNext][Y] == 0 and System[X][yNext] == 0: # Right and down free
                        reward = -2
                    elif System[xNext][Y] != 0 and System[X][yNext] != 0: # All surrounded
                        reward = -2
                    elif System[xNext][Y] != 0 and System[X][yNext] == 0: # Only forward free
                        reward = -2
                    else:
                        reward = -2                                      # Only down free

                    if log == True:                    
                        print("    Site occupied: Attempt to hop up")
                        print("    instant reward", reward)

            if JumpTransverse == 1: # hop down
                if System[xNext][Y] == 0: # below site free
                    System[X][Y], System[xNext][Y] = System[xNext][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xNext][Y] = JumpRateGrid[xNext][Y], JumpRateGrid[X][Y]
                    Transv_step = -1
                    
                    # Rewards from checking surroundings
                    if System[xPrev][Y] == 0 and System[X][yNext] == 0: # All free
                        reward = 0
                    elif System[xPrev][Y] != 0 and System[X][yNext] != 0: # Only down free
                        reward = 0      
                    elif System[xPrev][Y] != 0 and System[X][yNext] == 0: # Only forward and down free
                        reward = 0                                               
                    else:
                        reward = 0                                     # Only up free

                    # Update of coordinates 
                    X = xNext

                    if log == True:
                        print("    Hop down")
                        print("    instant reward", reward)

                else:  # below site occupied
                    if System[xPrev][Y] == 0 and System[X][yNext] == 0: # Right and up free
                        reward = -2
                    elif System[xPrev][Y] != 0 and System[X][yNext] != 0: # All surrounded
                        reward = -2       
                    elif System[xPrev][Y] != 0 and System[X][yNext] == 0: # Only forward free
                        reward = -2                                               
                    else:
                        reward = -2                                     # Only up free

                    if log == True:                                        
                        print("    Site occupied: Attempt to hop down")
                        print("    instant reward", reward)


                                 
           # Clustering reward
            # Update of Periodic boundary conditions
            xPrev = Ly - 1 if X == 0 else X - 1        
            xNext = 0 if X == Ly - 1 else X + 1
            yNext = 0 if Y == Lx - 1 else Y + 1
            yPrev = Lx - 1 if Y == 0 else Y - 1

            xPPrev = (X - 2) % Ly
            xNNext = (X + 2) % Ly
            yPPrev = (Y - 2) % Lx
            yNNext = (Y + 2) % Lx

            d_c = 1 # Equilibrium distance
            K = 1 # LJ constant
            r = 1/d_c # Nearest neighbours

            Dfront,Dback,Dup,Ddown = 0, 0, 0, 0
            DSfront,DSback,DSup,DSdown = 0, 0, 0, 0
            LJfront,LJback,LJup,LJdown = 0, 0, 0, 0
            fast_count = 0
            slow_count = 0

            # # Nearest neighbours 
            if log == True:            
                print(System)          
            if System[X][yNext] != 0:
                if log == True:                
                    print('forward occupied')
                    print(System[X][yNext])
                Dfront = abs(JumpRateGrid[X][yNext] - JumpRate)
                #LJfront = 1/r**2 - 1/r

                if JumpRateGrid[X][yNext] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yNext] != 0:
                    slow_count += 1 

            if System[xNext][yNext]!= 0:
                if log == True:                
                    print('forward-down occupied')
                    print(System[xNext][yNext])
                Dfrontdown = abs(JumpRateGrid[xNext][yNext]- JumpRate)
                #LJfrontdown = 1/r**2 - 1/r

                if JumpRateGrid[xNext][yNext] == 1:
                    fast_count += 1
                elif JumpRateGrid[xNext][yNext] != 0:
                    slow_count += 1 

            if System[xNext][Y] != 0:
                if log == True:                
                    print('down occupied')
                    print(System[xNext][Y])                
                Ddown = abs(JumpRateGrid[xNext][Y] - JumpRate)
                #LJdown = 1/r**2 - 1/r

                if JumpRateGrid[xNext][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xNext][Y] != 0:
                    slow_count += 1   

            if System[xNext][yPrev]!= 0:
                if log == True:                
                    print('back-down occupied')
                    print(System[xNext][yPrev])
                Dbackdown = abs(JumpRateGrid[xNext][yPrev]- JumpRate)
                #LJbackdown = 1/r**2 - 1/r

                if JumpRateGrid[xNext][yPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[xNext][yPrev] != 0:
                    slow_count += 1                                                       

            if System[X][yPrev] != 0:
                if log == True:                
                    print('back occupied')
                    print(System[X][yPrev])                
                Dback = abs(JumpRateGrid[X][yPrev] - JumpRate)
                #LJback = 1/r**2 - 1/r

                if JumpRateGrid[X][yPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yPrev] != 0:
                    slow_count += 1

            if System[xPrev][yPrev]!= 0:
                if log == True:                
                    print('back-up occupied')
                    print(System[xPrev][yPrev])
                Dbackup = abs(JumpRateGrid[xPrev][yPrev]- JumpRate)
                #LJbackup  = 1/r**2 - 1/r

                if JumpRateGrid[xPrev][yPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[xPrev][yPrev] != 0:
                    slow_count += 1   

            if System[xPrev][Y] != 0:
                if log == True:                
                    print('up occupied')
                    print(System[xPrev][Y])                

                Dup = abs(JumpRateGrid[xPrev][Y] - JumpRate)   
                #LJup = 1/r**2 - 1/r

                if JumpRateGrid[xPrev][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xPrev][Y]  != 0:
                    slow_count += 1
            
            if System[xPrev][yNext]!= 0:
                if log == True:                
                    print('front-up occupied')
                    print(System[xPrev][yPrev])
                Dfrontup = abs(JumpRateGrid[xPrev][yPrev]- JumpRate)
                #LJfrontup = 1/r**2 - 1/r

                if JumpRateGrid[xPrev][yPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[xPrev][yPrev] != 0:
                    slow_count += 1  


            # Second nearest neighbours                           
            if System[X][yNNext] != 0:
                if log == True:                
                    print('second forward occupied')
                    print(System[X][yNNext])
                DSfront = abs(JumpRateGrid[X][yNNext] - JumpRate)
                #LJfront = 1/r**2 - 1/r

                if JumpRateGrid[X][yNNext] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yNNext] != 0:
                    slow_count += 1 

            if System[X][yPPrev] != 0:
                if log == True:                
                    print('second back occupied')
                    print(System[X][yPPrev])                
                DSback = abs(JumpRateGrid[X][yPPrev] - JumpRate)
                #LJback = 1/r**2 - 1/r

                if JumpRateGrid[X][yPPrev] == 1:
                    fast_count += 1
                elif JumpRateGrid[X][yPPrev] != 0:
                    slow_count += 1

            if System[xNNext][Y] != 0:
                if log == True:                
                    print('second down occupied')
                    print(System[xNNext][Y])                
                DSdown = abs(JumpRateGrid[xNNext][Y] - JumpRate)
                #LJdown = 1/r**2 - 1/r

                if JumpRateGrid[xNNext][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xNNext][Y] != 0:
                    slow_count += 1                   

            if System[xPPrev][Y] != 0:
                if log == True:                
                    print('second up occupied')
                    print(System[xPPrev][Y])                

                DSup = abs(JumpRateGrid[xPPrev][Y] - JumpRate)   
                #LJup = 1/r**2 - 1/r

                if JumpRateGrid[xPPrev][Y] == 1:
                    fast_count += 1
                elif JumpRateGrid[xPPrev][Y]  != 0:
                    slow_count += 1



            JumpRate_diff = (-1)*float(Dfront + Dback + Dup + Ddown)
            JumpRate_diff_sec = (-1)*float(DSfront + DSback + DSup + DSdown)

            #LJ_function = (-K)*int(LJfront + LJback + LJup + LJdown)
            LJ_function = 0
            #reward += (JumpRate_diff + JumpRate_diff_sec) + LJ_function

            # if log == True:
            #     print("Particle jump rate:", JumpRate)
            #     print("fast_count", fast_count)
            #     print("slow_count", slow_count)

            #     print("Dfront", Dfront)
            #     print("Dback", Dback) 
            #     print("Dup", Dup) 
            #     print("Ddown", Ddown) 

            #     print("    JumpRate_diff", JumpRate_diff)         
            #     #print("LJ_function", LJ_function)                   
            #     print("    instant reward after jr difference", reward)


            if JumpRate == 1: #Fast particle
                counting_reward = fast_count - slow_count
            else: #Slow particle
                counting_reward = -fast_count + slow_count
            
            # reward += counting_reward


            if log == True:
                print("    Counting_reward", counting_reward)   

                print("    instant reward after counting numbers", reward)   

        else:
            if log == True:
                print("    Without velocity")
                
           # Initial surroundings check reward (aka second chance)
            if System[X][yNext] == 0:  
                if System[xNext][Y] == 0 and System[xPrev][Y] == 0: # If forward, up and down are free
                    reward = 1
                else:
                    reward = 0
            else:
                if System[xNext][Y] == 0 and System[xPrev][Y] == 0:
                    reward = 0
                else:  # If forward, up and down is occupied
                    reward = -1


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

        info["Along_Steps"] = Along_step
        info["Transv_Steps"] = Transv_step
                

        return observation, state, NN_input, self.reward, False, truncated, info
    