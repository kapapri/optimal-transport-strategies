# ==================================================
# Generation of the System and Utilities
# ==================================================

# Environment consisting on a (Lx*Ly) array as a grid with discrete sites that can have value 0 (empty) or 1 (occupied by an agent). It can start with a random distribution of n agents or a chess fashion filling homogeneously the lattice.

# Optimization: Crossing the right boundary and avoid the rest of agents with the least number of steps, so the rewards are distributed in order to get that result.
# Action space: TASEP is imposed so the posible actions are three: up, right, and down. Each episode lasts a constant number of movement attempts (max_steps).

# prob. distribution to check if it's learning 

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from numba import njit


# ==================================================
# Generation of the System and Utilities
# ==================================================

@njit
def trunc_gaussian_sample(mu, sigma):
    """
    It returns a sample of a gaussian, with mean mu and standard deviation sigma, only if it is in the [0, 1] interval
    """
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mu, sigma)
    return sample    

@njit
def truncated_gaussian(mu, sigma, size):
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = trunc_gaussian_sample(mu, sigma)   
    return samples

@njit
def checkboard(Lx, Ly, mu, sigma, n = False): 
    """ 
    Filling the lattice with particles alternatively (chess fashion).
    
    It returns two arrays of size (Lx*Ly):
    - System: it describes where the particles are with zeros (empty site) and ones (occupied site) in the lattice.
    - JumpRateGrid: Same structure as System but instead of ones where there is a particle, it has the corresponding jumping rates sampled from a truncated gaussian.
    """
    System = np.zeros((Ly, Lx), dtype=np.float32)  # Ly vectors with Lx (zero) components 
    System[::2, ::2] = 1  # Set even rows and even columns to 1
    System[1::2, 1::2] = 1  # Set odd rows and odd columns to 1
    
    JumpRateGrid = np.ones((Ly,Lx))*(-0.1) # Multiplied by -0.1 to impose the black color of the heatmap to this minimum
    for i in range(Ly):
        for j in range(Lx):
            if System[i][j] == 1:
                JumpRateGrid[i][j] = trunc_gaussian_sample(mu, sigma)
                    
    return System, JumpRateGrid

@njit
def random_system(Lx, Ly, mu, sigma, n): 
    """
    Filling the lattice (Lx*Ly) randomly with n particles
    
    It returns two arrays of size (Lx*Ly):
    - System: it describes where the particles are with zeros (empty site) and ones (occupied site) in the lattice.
    - JumpRateGrid: Same structure as System but instead of ones where there is a particle, it has the corresponding jumping rates sampled from a truncated gaussian.
    """
    assert n <= (Lx*Ly) # We cannot have more particles than avaliable sites
    System = np.concatenate((np.ones(n, dtype=np.float32), np.zeros(Lx*Ly-n, dtype=np.float32))) 
    np.random.shuffle(System)
    System = System.reshape((Ly, Lx))
    
    JumpRateGrid = np.ones((Ly,Lx), dtype=np.float32)*(-0.1)
    for i in range(Ly):
        for j in range(Lx):
            if System[i][j] == 1:
                JumpRateGrid[i][j] = trunc_gaussian_sample(mu, sigma) # probability of jumping forward or transversally           

    return System, JumpRateGrid





class LatticeTASEPEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, mode = "chess", Lx=4, Ly=2, n=4, max_steps = 4, mu = 0.5, sigma = 0.3): #Default values
        self.window_size = 512  # The size of the PyGame window
        
        # Choosing initial conditions
        if mode == "chess":
            mode = checkboard
        elif mode == "random":
            mode = random_system
            
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
        return self.System, self.JumpRateGrid

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
        self.System, self.JumpRateGrid = self.mode(self.Lx, self.Ly, self.mu, self.sigma, self.n)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation[0], info     
    
    def step(self, action):
        """
        It updates the arrays System and JumpRateGrid when the particle at [X, Y] lattice jumps once following TASEP 
        and counts the jump done after depending on the direction it is realized, along the main current (x-axis) or perpendicular (y-axis)

        """        
        Along_step = 0
        Transv_step = 0      
        Lx = self.Lx
        Ly = self.Ly
        self.step_counter += 1
        System = self.System.copy() # Apparently without that the initial observation updates every time a step is done
        JumpRateGrid = self.JumpRateGrid.copy()
        
        # Selection of a specific particle
        X = action // Lx
        Y = action - X * Lx       
        
        """ 
        The lattice is labelled from the left top to the right 
         as: action = 0, 1, 2    or    [X,Y]= (0,0), (0,1), (0,2) 
                      3, 4, 5                 (1,0), (1,1), (1,2)
        """        
    # TASEP, deciding direction: p=1/2 of jumping foward, p=1/4 of jumping up or down         
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

        # Updating the system
        if dice_jump < JumpRate:
            if JumpAlong == 0: # tries to hop forward
                if System[X][yNext] == 0:  # if next site is free
                    System[X][Y], System[X][yNext] = System[X][yNext], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[X][yNext] = JumpRateGrid[X][yNext], JumpRateGrid[X][Y]
                    Along_step = 1
                    self.reward += 100
                    

            if JumpTransverse == 0: # hop up
                if System[xPrev][Y] == 0: 
                    System[X][Y], System[xPrev][Y] = System[xPrev][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xPrev][Y] = JumpRateGrid[xPrev][Y], JumpRateGrid[X][Y]
                    Transv_step = 1
                    self.reward -= 500
                    

            if JumpTransverse == 1: # hop down
                if System[xNext][Y] == 0: 
                    System[X][Y], System[xNext][Y] = System[xNext][Y], System[X][Y]
                    JumpRateGrid[X][Y], JumpRateGrid[xNext][Y] = JumpRateGrid[xNext][Y], JumpRateGrid[X][Y]
                    Transv_step = -1
                    self.reward -= 500


        self.JumpRateGrid = JumpRateGrid
        self.System = System
            
        truncated = False
        if self.step_counter >= self.max_steps: # To stop the step after (max_steps) iterations
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