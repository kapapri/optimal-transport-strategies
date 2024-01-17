import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

# ==================================================
# Replay Memory/Replay Buffer
# ==================================================
# This is the list where the transitions are saved and randomly sampled to use more efficiently the experiences
# structure of the Q table
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Using a named tuple to be more readable, it creates a tuple named Transition with values (bla, blabla, blablabla)

# class that defines the Q table
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity) #list-like container with fast appends and pops on either end. # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def whole_list(self):
        return self.memory
    
    def fill(self, data):
        self.memory = data       
        return self.memory       

    def __len__(self):
        return len(self.memory) # Return the number of items in a list
    

# ==================================================
# DQN algorithm
# ==================================================
#   To calculate the current Q-value, the input is the state and the action.
#   Q-value = expected return of taking each action given the current state.

class DQN(nn.Module):
    # Según como sea el tipo de dato del state, tendremos que pasarlo a torch.tensor y modificar el shape de éste de forma que su size sea ([1, algo])
        # Si es un vector, solo hay que añadir state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Si es un int, state = torch.tensor(state, dtype=torch.float32).view(1,1)
        # Si es una matriz, se reduce la dimensionalidad del input state = torch.tensor(np.reshape(state, (1,16)), dtype=torch.float32)


    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)  # torch.nn.functional.linear(input_dims, weight_dims, bias=None): y=x*W+b
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

   
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)

def optimization(memory, policy_net, target_net, device, BATCH_SIZE, GAMMA, TAU, optimizer, loss_count):   
    # Perform one step of the optimization (on the policy network)
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        loss_count = 0
        return loss_count

# It samples a batch, concatenates all the tensors into a single one        
    transitions = memory.sample(BATCH_SIZE) 
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

#Check if this is necessary
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
                                                                                                                            #The map() function runs a lambda function over a list building a list-like collection of the results 
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    action_batch = action_batch.type(torch.int64)
    reward_batch = torch.cat(batch.reward)

# Compute Q(s_t, a)
    #  The model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken
    #  for each batch state according to policy_net

    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space (N), 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    #print(action_batch - 1)
    state_values = policy_net(state_batch)
    state_action_values = state_values.gather(1, action_batch - 1) # index = action_batch-1, so the indexes start at 0
                                                                # torch.gather(input, dim, index). Following the indexing structure (aling a specific dim), 
                                                                # it takes values of the input
    # For DQN_Check and Simple_Environment:
    #state_action_values = state_values.gather(1,action_batch) 

    # Compute the target network V(s_{t+1}) for all next states.
    #  Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
    #  This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad(): # # Evaluating the model with torch.no_grad() ensures that no gradients are computed 
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    #criterion = nn.L1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping (to prevent them from becoming too large)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    loss_sample = loss.item()
    loss_count += loss_sample
    
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
    
    return loss_count
