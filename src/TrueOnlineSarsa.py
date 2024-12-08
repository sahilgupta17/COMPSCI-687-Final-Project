from Environment import get_Cats_vs_monsters
import numpy as np
class TrueOnlineSarsa:
    def __init__(self, enviromnent, alpha, gamma, lambd, epsilon, num_episodes=100000):
        self.env = enviromnent
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon=epsilon
        self.num_episodes = num_episodes
        self.state_action_pairs = enviromnent.get_state_action_pairs()
        self.w = np.zeros(len(self.state_action_pairs))
        self.z = np.zeros(len(self.state_action_pairs))
    # feature function
    def x(self, state, action):
        feature = np.zeros(len(self.state_action_pairs))
        if state+(action,) in self.state_action_pairs:
            feature[self.state_action_pairs.index(state+(action,))] = 1
        return feature
        
    def run_sarsa(self):
        for _ in range(self.num_episodes):
            state = self.env.d0() # starting state
            action = self.env.get_next_action(state) # starting action
            x, Q_old= self.x(state,action), 0
            print(_)
            while True:
                next_state = self.env.get_next_state(state, action)
                if self.env.is_terminal(state):
                    break 
                reward = self.env.get_reward(state, action, next_state)
                next_action = self.env.get_next_action(next_state)
                x_prime = self.x(next_state,next_action)
                q,q_prime = self.w.T.dot(x), self.w.T.dot(x_prime)
                delta = reward + self.gamma*q_prime - q
                self.z = self.gamma * self.lambd * self.z + (1 - self.alpha * self.gamma * self.lambd * self.z.T.dot(x)) * x
                self.w = self.w + self.alpha*(delta + q - Q_old)*self.z - self.alpha*(q-Q_old)*x
                Q_old, x, action, state = q_prime, x_prime, next_action, next_state
        return Q_old

online_sarsa = TrueOnlineSarsa(enviromnent=get_Cats_vs_monsters(), alpha = 0.1,gamma = 0.99,lambd = 0.9,epsilon = 0.1,num_episodes = 500)
print(online_sarsa.run_sarsa())