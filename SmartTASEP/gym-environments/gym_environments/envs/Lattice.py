# Environment consisting on a (Lx*Ly) array as a grid with discrete sites that can have value 0 (empty) or 1 (occupied by an agent). It can start with a random distribution of n agents or a chess fashion filling homogeneously the lattice.

# Optimization: Crossing the right boundary and avoid the rest of agents with the least number of steps, so the rewards are distributed in order to get that result.
# Action space: TASEP is imposed so the posible actions are three: up, right, and down. Each episode lasts a constant number of movement attempts (max_steps).

# prob. distribution to check if it's learning 

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

def trunc_gaussian_jit_sample(mu, sigma):
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mu, sigma)
    return sample

def truncated_gaussian_jit(mu, sigma, size):
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = trunc_gaussian_jit_sample(mu, sigma)
        
    return samples


def generate_chess_map(Lx, Ly, mu, sigma, n = None):     # Generates two equivalent matrices representing the                                                                                                                              # lattice: One with the labels of the particles where 
                                                         # they are located (Map) and another with its jumping rates (JumpRateGrid)
    n = Lx * Ly // 2 # number of particles
    Labels = [x + 1 for x in range(n)]
    
    # Assigning jumping rates from a truncated gaussian distribution
    JumpRates = np.zeros(n)
    JumpRates = truncated_gaussian_jit(mu, sigma, n)
            
    # Associating Labels with its jumping rates in a list
    Particles = list(zip (Labels, JumpRates))

    # Checkboard map
    System = np.zeros((Ly, Lx), dtype=int)  # Ly vectors with Lx (zero) components 
    System[::2, ::2] = 1  # Set even rows and even columns to 1
    System[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    Map = np.zeros((Ly, Lx), dtype=int)  # Ly vectors with Lx (zero) components 
    
    # Jumping Rate map
    JumpRateGrid = np.ones((Ly, Lx), dtype=float)*(-0.001) # Initialized with -0.0001 for the movie
    k = 0
    for i in range(Ly):
        for j in range(Lx):
            if System[i][j] == 1:
                Map[i][j] = Labels[k]
                JumpRateGrid[i][j] = JumpRates[k] 
                k += 1
    
    
    # EXTRA: Different approach
    # Checkboard system with labels and jumping rates together
    
#     System = np.zeros((Ly, Lx, 2))  # Ly set of Lx vectors with 2 components 
#     counter = 0
#     for i in range(Ly):
#         if counter % 2 == 0:
#             for j in range(0, Lx, 2):
#                 System[i][j][0] = 1
#             for j in range(1, Lx, 2):
#                 System[i][j][0] = 0
#         else:
#             for j in range(0, Lx, 2):
#                 System[i][j][0] = 0
#             for j in range(1, Lx, 2):
#                 System[i][j][0] = 1
#         counter += 1
        
#     k = 0
#     for i in range(Ly):
#         for j in range(Lx):
#             if System[i][j][0] == 1:
#                 System[i][j] = Particles[k]
#                 k += 1
                
    return System, Map, JumpRateGrid

def generate_random_map(Lx, Ly, mu, sigma, n): # Generates two equivalent matrices representing the                                                                            # lattice: One with the labels of the particles where 
                                               # they are located (Map) and another with its jumping rates (JumpRateGrid)
    assert n <= (Lx*Ly) 
    
    Labels = [x+1 for x in range(n)]
    
    # Assigning jumping rates from a truncated gaussian distribution    
    JumpRates = np.zeros(n)
    JumpRates = truncated_gaussian_jit(mu, sigma, n)
            
    # Associating Labels with its jumping rates in a list
    Particles = list(zip (Labels, JumpRates))

    # Random map
    Map = np.concatenate((Labels, np.zeros(Lx*Ly-n, int)))
    np.random.shuffle(Map)
    Map = Map.reshape((Ly, Lx))
    
    # Jumping rate map
    JumpRateGrid = np.zeros((Ly, Lx))
    System = np.zeros((Ly, Lx))
    for i in range(Ly):
        for j in range(Lx):
            label = Map[i, j]
            if label != 0:
                # Find the corresponding JumpRate for the label
                JumpRateGrid[i, j] = Particles[label - 1][1]
                System[i, j] = 1
    
    ## EXTRA:
    #System = Map
    
    return System, Map, JumpRateGrid


class LatticeEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, mode = "chess", Lx=4, Ly=2, n=4, max_steps = 4, mu = 0.5, sigma = 0.3): #Default values
        self.window_size = 512  # The size of the PyGame window
        
        # Choosing initial conditions
        if mode == "chess":
            mode = generate_chess_map
        elif mode == "random":
            mode = generate_random_map
            
        self.mode = mode # Initial distribution of the map
        self.Lx = Lx  # The horizontal length
        self.Ly = Ly  # The vertical length
        self.n = n # number of elements
        
        self.mu = mu
        self.sigma = sigma
        
        self.max_steps = max_steps # number of movements in each iteration        
        self.step_counter = 0        
        self.reward = 0
        
        self.Along_count = 0
        self.Transv_count = 0        
        self.CurrentAlong = 0
        self.CurrentTransv = 0
 
        # Lx*Ly grid with 0 and 1 values corresponding to empty or occupied site
        self.observation_space = spaces.MultiBinary([Ly, Lx])      
        # e.g: array([[ 0, 0, 0, 0],
#                    [ 0, 0, 1, 0],
#                    [ 0, 0, 0, 1]]
        # Later we will have two Lx*Ly grids with the particle labels values or the jumping rates of each particle

        #self.observation_space = spaces.MultiBinary(Lx*Ly)      
    
        # We have n actions, corresponding to choosing a specific particle to move
        self.action_space = spaces.Discrete(n)  

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        return self.System, self.board, self.JumpRateGrid

    # Auxiliary information
    def _get_info(self):
        # return {"Along_Steps":[],"Transv_Steps":[]}
        return {}


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
        self.System, self.board, self.JumpRateGrid = self.mode(self.Lx, self.Ly, self.mu, self.sigma, self.n)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation[0], info     
    

    def step(self, action):
        Along_step = 0
        Transv_step = 0
        
        Lx = self.Lx
        Ly = self.Ly
        self.step_counter += 1
        
        System = self.System.copy()
        Map = self.board.copy() ####### CHECK if this is necessary: Apparently without that the initial observation updates every time a step is done
        JumpRateGrid = self.JumpRateGrid.copy() 
        
        """ 
        The lattice is labelled from the left top to the right 
         as Z= 0, 1, 2    or    [X,Y]= (0,0), (0,1), (0,2) 
               3, 4, 5                 (1,0), (1,1), (1,2)
        """   
        
        # Selection of a specific particle
        for i in range(Ly):
            for j in range(Lx):
                label = Map[i, j]
                if label == (action+1):
                    X = i
                    Y = j
        
        # Simple implementation of Periodic boundary conditions
        xPrev = Ly - 1 if X == 0 else X - 1        
        xNext = 0 if X == Ly - 1 else X + 1
        yNext = 0 if Y == Lx - 1 else Y + 1
        
        # Implementation of TASEP: p=1/2 of jumping foward, p=1/4 of jumping up or down  
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

        if dice_jump > JumpRate:
            if JumpAlong == 0:             # hop forward
                if Map[X][yNext] == 0:  # if next site is free
                    temp = Map[X][Y]
                    Map[X][Y] = Map[X][yNext]
                    Map[X][yNext] = temp

                    temp2 = JumpRateGrid[X][Y]
                    JumpRateGrid[X][Y] = JumpRateGrid[X][yNext]
                    JumpRateGrid[X][yNext] = temp2

                    temp3 = System[X][Y]
                    System[X][Y] = System[X][yNext]
                    System[X][yNext] = temp3

                    Along_step += 1
                    self.reward += 100

            if JumpTransverse == 0: # hop up
                if Map[xPrev][Y] == 0: 
                    temp = Map[X][Y]
                    Map[X][Y] = Map[xPrev][Y]
                    Map[xPrev][Y] = temp

                    temp2 = JumpRateGrid[X][Y]
                    JumpRateGrid[X][Y] = JumpRateGrid[xPrev][Y]
                    JumpRateGrid[xPrev][Y] = temp2

                    temp3 = System[X][Y]
                    System[X][Y] = System[xPrev][Y]
                    System[xPrev][Y] = temp3

                    Transv_step += 1
                    self.reward -= 500

            if JumpTransverse == 1: # hop down
                if Map[xNext][Y] == 0:  
                    temp = Map[X][Y]
                    Map[X][Y] = Map[xNext][Y]
                    Map[xNext][Y] = temp

                    temp2 = JumpRateGrid[X][Y]
                    JumpRateGrid[X][Y] = JumpRateGrid[xNext][Y]
                    JumpRateGrid[xNext][Y] = temp2

                    temp3 = System[X][Y]
                    System[X][Y] = System[xNext][Y]
                    System[xNext][Y] = temp3                

                    Transv_step -= 1
                    self.reward -= 500


        self.board = Map ####### CHECK if this is necessary
        self.JumpRateGrid = JumpRateGrid
        self.System = System
            
        truncated = False
        if self.step_counter >= self.max_steps:
            truncated = True
                
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        
        info["Along_Steps"] = Along_step
        info["Transv_Steps"] = Transv_step


        return observation[0], self.reward, False, truncated, info
    
    

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

#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode(
#                 (self.window_size, self.window_size)
#             )
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
#         pix_square_size = (
#             self.window_size / self.size
#         )  # The size of a single grid square in pixels

#         # First we draw the target
#         pygame.draw.rect(
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )
#         # Now we draw the agent
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (self._agent_location + 0.5) * pix_square_size,
#             pix_square_size / 3,
#         )

#         # Finally, add some gridlines
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (0, pix_square_size * x),
#                 (self.window_size, pix_square_size * x),
#                 width=3,
#             )
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self.window_size),
#                 width=3,
#             )

#         if self.render_mode == "human":
#             # The following line copies our drawings from `canvas` to the visible window
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()

#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to keep the framerate stable.
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )


    #-------------------------------------------------------------------------------------------------------------------------------------------------------    

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()