import numpy as np
from itertools import product
import random
class Environment:
    # Function that initialzes the required things in the Cat vs Monster class
    def __init__(self, gamma, grid_shape,terminal_states, actions, illegal_states, reward_func, s0=None) -> None:
        '''
        Parameters:
            gamma(float): future reward discount
            grid_shape(tuple(int)): the shape of the grid world
            terminal_states(list(tuple(int))): the ids of the terminal states
            actions(list(str)): list of actions
            illegal_states(list(tuple(int))): ids of the states where agent is not allowed to enter
            reward_func(state, action, next_state): a function that takes in the reward and action and returns the reward for state_next
            s0(dict(tuple:prob) or tuple): possible start state(s)
        User Guide:
            self.v is the current value function accociated with the domain
            self.actions are actions the agent can take
            self.q is an np array which is a action lengthed array for each state
            self.state_ids is a dict with all the legal possible states where the agent can be in or a single state
        '''
        self.v = np.zeros(grid_shape) # initialize v to zeros
        self.actions = actions # array of the possible actions
        self.gamma = gamma
        self.solid_furniture = illegal_states
        self.terminal_states = terminal_states
        self.rewards = reward_func
        self.q = np.zeros( grid_shape + (len(actions),)) # initialize q to zeros
        if type(s0) is dict: 
            self.s0 = s0
        elif s0 != None:
            self.s0 = {s0:1}
        else:
            self.s0 = s0
        
        self.state_ids = list(product(*[range(d) for d in grid_shape]))
        for i in self.solid_furniture + self.terminal_states:
            self.state_ids.remove(i)    

    def d0(self, anywhere = False):
        if anywhere or self.s0 == None:
            return self.state_ids[np.random.choice(np.arange(len(self.state_ids)))]
        return random.choices(list(self.s0.keys()), weights=list(self.s0.values()), k=1)[0]

def rewards(state, action, state_next):
    rewards = np.array([-0.05 for _ in range(25)]).reshape(5,5) # array with all the rewards set to -0.5
    rewards[4,4], rewards[0,3], rewards[4,1], rewards[2,1], rewards[2,2], rewards[2,3], rewards[3,2] = 10, -8, -8, 0,0,0,0 # updating the reward for the final state and the monster state
    return rewards[state_next[0],state_next[1]]
environment = Environment(gamma=0.925, grid_shape=(5,5), terminal_states=[(4,4)],actions=['AU', 'AR', 'AD', 'AL'], illegal_states=[(2,1),(2,2),(2,3),(3,2)], reward_func=rewards, s0 = (0,0))
print(environment.d0())
                