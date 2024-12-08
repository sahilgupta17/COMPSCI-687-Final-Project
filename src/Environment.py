import numpy as np
from itertools import product
import random

class Environment:
    # Function that initialzes the required things in the Cat vs Monster class
    def __init__(self, gamma, grid_shape, terminal_states, actions, reward_func=None, action_function = None, illegal_states=[], s0=None, next_state_probs = None) -> None:
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
        self.reward_function = reward_func
        self.q = np.zeros( grid_shape + (len(actions),)) # initialize q to zeros
        
        self.policy = np.zeros(grid_shape + (len(actions),))
        self.policy.fill(0.25)
        
        # setup for the starting state
        if type(s0) is dict: 
            self.s0 = s0
        elif s0 != None:
            self.s0 = {s0:1}
        else:
            self.s0 = s0
        
        self.state_ids = list(product(*[range(d) for d in grid_shape]))
        for i in self.solid_furniture + self.terminal_states:
            self.state_ids.remove(i)  
        if action_function != None: 
            self.action_function = action_function
        else:
            # Selects action based on a greedy epsilon stratergy
            def next_action(state, policy):
                action_list = policy[state[0], state[1]]
                return np.random.choice(len(action_list), p=action_list)
            
            self.action_function = next_action
        self.next_state_probs = next_state_probs
        
    # function to give the starting state of the run
    def d0(self, anywhere = False):
        if anywhere or self.s0 == None:
            return self.state_ids[np.random.choice(np.arange(len(self.state_ids)))]
        return random.choices(list(self.s0.keys()), weights=list(self.s0.values()), k=1)[0]
    
    # function to give the next action based on the current state
    def get_next_action(self,state):
        return self.action_function(state,self.policy)

    # function to give the reward of taking action a and going from state s to s'
    def get_reward(self,state,action,next_state):
        return self.reward_function(state,action,next_state)

    # function to return the next state
    def get_next_state(self,state,action):
        next_state_probs = self.next_state_probs(state, self.actions[action])
        state_keys, state_values = list(next_state_probs.keys()), list(next_state_probs.values())
        state_next = state_keys[np.random.choice(len(state_keys), p=state_values)]
        return state_next

    def get_Q(self):
        return self.q
    
    def update_Q(self,q):
        self.q = q

    def get_policy(self):
        return self.policy
    
    def update_policy(self,policy):
        self.policy = policy

    def get_num_states_and_actions(self):
        return len(self.state_ids), len(self.actions)

    def get_state_action_pairs(self):
        pairs = []
        for state in self.state_ids:
            for action in range(len(self.actions)):
                pairs.append(state + (action,))
        return pairs

    def is_terminal(self,state):
        return state in self.terminal_states
    
def get_Cats_vs_monsters():
    # next_ state's probability for cvm
    def next_state_probs_cvm(s, a):
        possible_states = {}
        
        # helper function to check if a state is valid
        def is_valid(state):
            i, j = state
            return 0 <= i < 5 and 0 <= j < 5 and state not in [(2,1),(2,2),(2,3),(3,2)]

        # define the target state based on action
        target_state = {
            'AU': (s[0] - 1, s[1]),  
            'AD': (s[0] + 1, s[1]),  
            'AL': (s[0], s[1] - 1),  
            'AR': (s[0], s[1] + 1)}
        
        temp = s # set temp to the default state
        if is_valid(target_state[a]): # check if the new possible state is a valid state
            temp = target_state[a]
        possible_states[temp] = possible_states.get(temp, 0) + 0.7 # move into the new state 

        confused_actions = {
            'AU': [(s[0], s[1] - 1), (s[0], s[1] + 1)],
            'AD': [(s[0], s[1] + 1), (s[0], s[1] - 1)],
            'AL': [(s[0] - 1, s[1]), (s[0] + 1, s[1])],
            'AR': [(s[0] + 1, s[1]), (s[0] - 1, s[1])]}
        
        for confused_state in confused_actions[a]:
            if is_valid(confused_state):
                possible_states[confused_state] = possible_states.get(confused_state, 0) + 0.12
            else:
                possible_states[s] = possible_states.get(s, 0) + 0.12
        
        # account for staying in the same place
        possible_states[s] = possible_states.get(s, 0) + 0.06  # Adding the residual probability

        return possible_states
    # rewards for cats_vs_monster
    def rewards_cvm(state, action, state_next):
        rewards = np.array([-0.05 for _ in range(25)]).reshape(5,5) # array with all the rewards set to -0.5
        rewards[4,4], rewards[0,3], rewards[4,1], rewards[2,1], rewards[2,2], rewards[2,3], rewards[3,2] = 10, -8, -8, 0,0,0,0 # updating the reward for the final state and the monster state
        return rewards[state_next[0],state_next[1]]

    return Environment(gamma=0.925, grid_shape=(5,5), terminal_states=[(4,4)],actions=['AU', 'AR', 'AD', 'AL'], illegal_states=[(2,1),(2,2),(2,3),(3,2)], s0 = (0,0), reward_func=rewards_cvm, next_state_probs=next_state_probs_cvm)

# environment = get_Cats_vs_monsters()
# s0 = environment.d0()
# action = environment.get_next_action(s0)
# s_prime = environment.get_next_state(s0,action)
# reward = environment.get_reward(s0,action,s_prime)
# print(action)            
# print(s_prime)
# print(reward)