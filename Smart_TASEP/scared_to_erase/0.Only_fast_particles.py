import math
import numpy as np
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML

from tqdm import tqdm # for the interactive mode this can be annoying

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# structure of the Q table
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that defines the Q table
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)    

def get_state_training(system, L):
    state = np.zeros(L*L)
    for x in range(L):
        for y in range(L):
            index = x*L + y
            value = system[x][y]
            state[index] = value

    return state

def get_state(system, X, Y, L, newL):
    state = np.zeros(L*L)
    for i in range(L):
        x_index = X - int(L/2) + i
        if x_index > newL-1:
            x_index += -newL
        elif x_index < 0:
            x_index += newL

        for j in range(L):
            y_index = Y - int(L/2) + j
            if y_index > newL-1:
                y_index += -newL
            elif y_index < 0:
                y_index += newL

            index = i*L + j
            value = system[x_index][y_index]
            state[index] = value

    return state

def get_coordinates_from_patch(x, y, X, Y, L, newL):
    """
    Convert coordinates of a point within the patch to corresponding coordinates in the system.
    
    Args:
    x, y: Coordinates of the point within the patch.
    X, Y: Coordinates of the center of the patch in the system.
    L: Size of the patch.
    newL: Size of the system.
    
    Returns:
    Tuple of (x_sys, y_sys): Coordinates of the corresponding point in the system.
    """
    # Calculate the index offsets from the center of the state
    x_offset = x - int(L / 2)
    y_offset = y - int(L / 2)
    
    # Calculate the corresponding coordinates in the system
    x_sys = X + x_offset
    y_sys = Y + y_offset
    
    # Apply toroidal boundary conditions
    if x_sys < 0:
        x_sys += newL
    elif x_sys >= newL:
        x_sys -= newL
    
    if y_sys < 0:
        y_sys += newL
    elif y_sys >= newL:
        y_sys -= newL
    
    return x_sys, y_sys    

def select_action_training(state): 
    global steps_done # count total number of steps to go from almost random exploration to more efficient actions
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # exploitation
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
    else:
        # select a random action; 
        rand_action = random.randint(0,L*L-1) # random lattice site in the observation patch
        return torch.tensor([[rand_action]], device=device, dtype=torch.long)

def select_action_post_training(state): 
    # interpret Q values as probabilities when simulating dynamics of the system 
    # in principle this could be easily extended to make this more general, but i am a lazy boi
    with torch.no_grad():
        # print("state ", state)
        Q_values = trained_net(state)
        # print("Q-values ", Q_values)
        probs = torch.softmax(Q_values, dim=1) # converts logits to probabilities (torch object)
        # print("probs ", probs)
        dist = Categorical(probs) # feeds torch object to generate a list of probs (numpy object ?)
        # print("dist ", dist)
        action = dist.sample().numpy()[0] # sample list of probs and return the action

        return action

# move
def step(lattice, X, Y, L, log = False):
    newX = -1
    newY = -1
    # periodic boundaries
    nextX = X + 1 if X < L - 1 else 0
    nextY = Y + 1 if Y < L - 1 else 0
    prevY = Y - 1 if Y > 0 else L - 1
    # update position
    direction = random.randint(0,3)
    if direction == 0 or direction == 1: # jump to the right
        newX = nextX
        newY = Y
    elif direction == 2: # jump to the top
        newY = nextY
        newX = X
    else: # jump to the bottom
        newY = prevY
        newX = X

    current_along = 0
    reward = 1 # simply for choosing a particle and not an empty space
    if lattice[newX][newY] == 0: # free next site
        lattice[X][Y] = 0
        lattice[newX][newY] = 1
        if log == True:
            print("  jump done")        
        if newX != X: # we have jump forward
            current_along = 1
            reward += 10
            if log == True:
                print("  moved forward")
        else:
            if log == True:
                print("  moved up or down") 
    else:
        if log == True:
            print("  it couldn't jump :(")                         
            
    # surroundings reward
    reward += int(-1*(lattice[X][prevY] + lattice[X][nextY]) - 1*(2*lattice[nextX][Y] - 1))

    next_state = get_state_training(lattice, L)

    return reward, next_state, current_along

def optimize_model():
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        return
    transitions = memory.sample(BATCH_SIZE) # draws a random set of transitions; the next_state for terminal transition will be NONE
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # turn [transition, (args)] array into [[transitions], [states], [actions], ... ]

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s != None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space, 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch) # input = policy_net, dim = 1, index = action_batch

    # Compute Q^\pi(s_t,a) values of actions for non_final_next_states by using target_net (old policy_net), from which max_a{Q(s_t, a)} are selected with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # target_net produces a vector of Q^pi(s_t+1,a)'s and max(1)[0] takes maxQ
    # Compute the expected Q^pi(s_t,a) values for all BATCH_SIZE (default=128) transitions
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def do_training(num_episodes, L, density, Nt, log = False):
    # the training is done in squared patches of the lattice
    for i_episode in tqdm(range(num_episodes)):
        # start with random initial conditions
        N = int(L*L*density) 
        lattice = np.zeros(shape=(L,L))
        n = 0
        while n < N:
            X = random.randint(0, L-1)
            Y = random.randint(0, L-1)
            if lattice[X][Y] == 0:
                lattice[X][Y] = 1
                n += 1
        if log == True:
            print("initial lattice", lattice)

        # main update loop; I use Monte Carlo random sequential updates here
        score = 0
        total_current = 0
        selected_empty_site = 0
        for t in range(Nt):
            for i in range(L*L):                
                state = get_state_training(lattice, L) # the patch
                if log == True:
                    print("state before", state)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = select_action_training(state) # get the index of the particle
                lattice_site = action.item() # a number, and we encode it as x*L + y
                selectedX = int(lattice_site / L)
                selectedY = int(lattice_site % L)
                if log == True:
                    print("- lattice_site:", lattice_site)
                    print("- selectedX:", selectedX)  
                    print("- selectedY:", selectedY)                                                      
                if lattice[selectedX][selectedY] != 0:
                    if log == True:
                        print("\n there is a particle")                    
                    reward, next_state, current_along = step(lattice, selectedX, selectedY, L, log) # update particle's position and do stochastic part
                    total_current += current_along / (L*L*Nt)
                    reward = torch.tensor([reward], device=device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) 
                    if log == True:
                        print(" state after ", next_state)
                    memory.push(state, action, next_state, reward)  
                else:
                    if log == True:
                        print("\n empty site chosen")

                    reward = -5  
                    selected_empty_site +=1
                    reward = torch.tensor([reward], device=device)  
                    memory.push(state, action, state, reward)      

                score += reward

            optimize_model()
            # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ)θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            

        # print("Training episode ", i_episode, " is over. Current = ", total_current, "; Selected empty sites / L*L = ", selected_empty_site / (Nt*L*L))
        rewards.append(score) 
        current.append(total_current)
        plot_score() # here if you want to see the training
                     # only with interactive python

    torch.save(target_net.state_dict(), PATH)
    plot_score(show_result=True) # here to see the result
    plt.ioff()
    # plt.show() # uncomment to see training
    plt.savefig("./Training_Reward.png", format="png", dpi=600) # only withOUT interactive python

# plots
def plot_score(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() # clf -- clear current figure
        plt.title('Training...')
    plt.xlabel('Episode duration')
    plt.ylabel('Reward')
    plt.plot(rewards)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            output = "./training_score.png"
            plt.savefig(output, format = "png", dpi = 300)

def plot_current():
    plt.figure(2)
    plt.xlabel('Episode duration')
    plt.ylabel('Average current over runs')
    plt.plot(current)

# Animation
def create_animation(Frames_movie):
    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    cv0 = Frames_movie[0]
    im = ax.imshow(cv0, cmap="gnuplot")
    cb = fig.colorbar(im, cax=cax)
    
    tx2 = ax.set_title('Frame 0 after one MCS of the last run', y=1)
    
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

# Main
if __name__ == '__main__':
    Jessie_we_need_to_train_NN = False
    Post_training = True
    log = False
    ############# Model parameters for Machine Learning #############
    num_episodes = 200      # number of training episodes
    BATCH_SIZE = 200        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.001         # EPS_END is the final value of epsilon
    EPS_DECAY = 100         # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-3               # LR is the learning rate of the AdamW optimizer
    ############# Lattice simulation parameters #############
    L = 5                   # Squared patches for the training
    density = 0.5 # work with half-density
    N = L*L*density
    Nt = 100               # episode duration
    n_observations = L*L    # just give network a difference between positive and negative spins
    n_actions = L*L         # patch size, in principle, the empty spots can also be selected
    hidden_size = 128        # hidden size of the network
    PATH = "./2d_TASEP_NN_params.txt"

    ############# Do the training if needed ##############
    if Jessie_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(100*Nt) # the overall memory batch size 
        #memory_prey = ReplayMemory(Nt) # the overall memory batch size 
        rewards = []
        current = []
        steps_done = 0
        
        do_training(num_episodes, L, density, Nt, log) 

    ############# Post-training simulation ##############
    if Post_training:
        runs = 10
        newL = 5
        Nt = 500
        trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
        trained_net.load_state_dict(torch.load(PATH))

        complete_short_movie = np.zeros((runs, Nt + 1, newL, newL))  # Array of episodes frames for each run
        one_short_movie = np.zeros((Nt + 1, newL, newL))  # Frames of set of movements after one episode

        current = np.zeros(Nt)
        for run in tqdm(range(runs)):
            # start with random initial conditions
            N = int(newL*newL*density) 
            lattice = np.zeros(shape=(newL,newL))
            n = 0
            while n < N:
                X = random.randint(0, newL-1)
                Y = random.randint(0, newL-1)
                if lattice[X][Y] == 0:
                    lattice[X][Y] = 1
                    n += 1

            one_short_movie[0] = lattice

            for t in range(Nt):
                total_current = 0
                for i in range(newL*newL):
                   # Random sampling of the new lattice to apply the training
                    X = random.randint(0, newL-1)
                    Y = random.randint(0, newL-1)
                    state = get_state(lattice, X, Y, L, newL)
                    
                   # to check how does the simulation perform for the
                   # patch with the same size than the system and sampling always from the center
                    # uncomment line below and comment the three lines above
                    # state = get_state_training(lattice, L)

                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = select_action_post_training(state)
                    lattice_site = action.item() # a number, and we encode it as x*L + y

                    # for get_state_training
                    # selectedX = int(lattice_site / L)
                    # selectedY = int(lattice_site % L)                    

                    # for get_state
                    patchX = int(lattice_site / L)
                    patchY = int(lattice_site % L)
                    selectedX, selectedY = get_coordinates_from_patch(patchX, patchY, X, Y, L, newL)

                    # to check how does the simulation perform for the random "stupid" simulation, 
                    # uncomment two lines below and comment the two lines above
                    #selectedX = random.randint(0, L-1)
                    #selectedY = random.randint(0, L-1)
                    #print("picked lattice site ", lattice[selectedX][selectedY])
                    if lattice[selectedX][selectedY] != 0:
                        newX = -1
                        newY = -1
                        nextX = selectedX + 1 if selectedX < newL - 1 else 0
                        nextY = selectedY + 1 if selectedY < newL - 1 else 0
                        prevY = selectedY - 1 if selectedY > 0 else newL - 1
                        # update position
                        direction = random.randint(0,3)
                        if direction == 0 or direction == 1: # jump to the right
                            newX = nextX
                            newY = selectedY
                        elif direction == 2: # jump to the top
                            newY = nextY
                            newX = selectedX
                        else: # jump to the bottom
                            newY = prevY
                            newX = selectedX

                        if lattice[newX][newY] == 0:
                            lattice[selectedX][selectedY] = 0
                            lattice[newX][newY] = 1
                            if newX != selectedX: # we have jump forward
                                total_current += 1.0/(newL*newL*runs)
                                #print("jumped!\n")
                    #else:
                        #print("ALARM! ALARM!")

                    one_short_movie[t + 1] = lattice

                current[t] += total_current #sum of the currents of all runs

            complete_short_movie[run] = one_short_movie

        plot_current()
        plt.savefig("./Post_Current.png", format="png", dpi=600) # only withOUT interactive python

        average_current = 0
        for t in range(Nt):
            average_current += current[t] / Nt

        print("average current = ", average_current)

        filename = "2d_TASEP_current_" + str(newL) + "_runs" + str(runs) + ".txt"
        with open(filename, 'w') as f:
            for t in range(Nt):
                output_string = str(t) + "\t" + str(current[t]) + "\n"
                f.write(output_string)            


        # Movie
        ani = create_animation(complete_short_movie[runs-1])
        HTML(ani.to_jshtml()) # interactive python
        ani.save("./Movie"+ str(newL) + "_runs" + str(runs) + ".mp4")