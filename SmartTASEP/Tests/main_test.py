import torch
from Environment import LatticeTASEPEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random policy implementation (no learning)
def select_particle_random(env):
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)    

def maintest():
 # Hyperparameters for environment
    env_params = {
        'render_mode': None,
        'mode': "chess",        
        'Lx': 8,
        'Ly': 1,
        'N': 4,
        'max_steps': 4,
        'mu': 0.5, 
        'fixed_sigma': 0.001                       
    }    
    # Initialize environment
    env = LatticeTASEPEnv(env_params)

    state, info = env.reset()
    
    action = select_particle_random(env)
    observation, reward, terminated, truncated, info = env.step(action)
    next_state = observation
    
    print("state 1", state)
    print("action 1", action)
    print("reward 1", reward)    
    print("next state 1", observation)
    
    action2 = select_particle_random(env)
    observation2, reward2, terminated2, truncated2, info2 = env.step(action2)
    
    print("state 2", observation)
    print("action 2", action2)
    print("reward 2", reward2)        
    print("next state 2", observation2)
    
    action3 = select_particle_random(env)
    observation3, reward3, terminated3, truncated3, info3 = env.step(action3)
    
    print("state 3", observation2)
    print("action 3", action3)
    print("reward 3", reward3)        
    print("next state 3", observation3)

if __name__ == '__main__':
    maintest()