import numpy as np # type: ignore
import random # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Constants
ACTIONS = ["AU", "AR", "AD", "AL"]
ACTIONS_TO_SYMBOLS = {
    "AU": "\u2191", 
    "AD": "\u2193", 
    "AL": "\u2190", 
    "AR": "\u2192"
}
GOAL_STATES=[(4, 4)] # only state 21 is the goal state
TERMINAL_STATES = [(4, 4)] # only state 21 is terminal.
OBSTACLES = [(2, 1), (2, 2), (2, 3), (3, 2)] # locations of where obstacles are present.
MONSTER_STATES = [(0, 3), (4, 1)] # locations of where monster is present.
NUM_ROWS, NUM_COLS, NUM_STATES = 5, 5, 25
OPTIMAL_POLICY = np.array(["AR", "AD", "AL", "AD", "AD", "AR", "AR", "AR", "AR", "AD", "AU", None, None, None, "AD", "AU", "AL", None, "AR", "AD", "AU", "AR", "AR", "AR", "AU"])
OPTIMAL_POLICY_STATE_VALUES = np.array([2.6638, 2.9969, 2.8117, 3.6671, 4.8497, 2.9713, 3.5101, 4.0819, 4.8497, 7.1648, 2.5936, 0, 0, 0, 8.4687, 2.0992, 1.0849, 0, 8.6097, 9.5269, 1.0849, 4.9465, 8.4687, 9.5269, 0])

class TrueOnlineSARSA_CatVsMonsters:
    def __init__(self, gamma, alpha, epsilon, lambda_, num_rows, num_cols):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = self.num_rows * self.num_cols
        self.Q = np.zeros((self.num_states, len(ACTIONS))) 
        self.E = np.zeros((self.num_states, len(ACTIONS)))  
        self.V = np.zeros(self.num_states)  
        self.policy = np.random.choice(ACTIONS, self.num_states)  
        self.state_action_pairs = [(s, a) for s in range(self.num_states) for a in range(len(ACTIONS))]
    
    def get_state_idx(self, state):
        return state[0] * self.num_cols + state[1]
    
    def get_next_state(self, curr_state, action):
        movement_directions = {
            "AU": (-1, 0),  # Up
            "AD": (1, 0),   # Down
            "AL": (0, -1),  # Left
            "AR": (0, 1)    # Right
        }
        
        if curr_state in TERMINAL_STATES:
            return curr_state
        
        random_num = np.random.rand()
        actual_movement_direction = action
        
        if random_num <= 0.70:
            # Cat moves in desired direction
            actual_movement_direction = action
        elif 0.7 < random_num <= 0.82:
            # Cat moves 90 degrees to the right action
            actual_movement_direction = ACTIONS[(ACTIONS.index(action) + 1) % 4]
        elif 0.82 < random_num <= 0.94:
            # Cat moves 90 degrees to the left action
            actual_movement_direction = ACTIONS[(ACTIONS.index(action) - 1) % 4]
        else:
            return curr_state
        
        next_state = (curr_state[0] + movement_directions[actual_movement_direction][0], 
                      curr_state[1] + movement_directions[actual_movement_direction][1])
        
        return next_state if self.is_valid_state(next_state) else curr_state
    
    def is_valid_state(self, state):
        return state not in OBSTACLES and state[0] >= 0 and state[0] < self.num_rows and state[1] >= 0 and state[1] < self.num_cols

    def get_reward(self, curr_state, next_state):
        if curr_state in TERMINAL_STATES and next_state in TERMINAL_STATES:
            return 0
        elif curr_state not in GOAL_STATES and next_state in GOAL_STATES:
            return 10
        elif next_state in MONSTER_STATES:
            return -8
        return -0.05

    def get_initial_state(self):
        invalid_state_idxs = {self.get_state_idx((row, col)) for row, col in OBSTACLES}
        invalid_state_idxs.add(self.get_state_idx((4, 4)))  # Goal state is terminal
        valid_states = [state for state in range(self.num_states) if state not in invalid_state_idxs]
        state_idx = np.random.choice(valid_states)
        row, col = state_idx // self.num_cols, state_idx % self.num_cols
        return (row, col)

    def take_next_action(self, state_idx):
        # state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
        q_values = {action: self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS}
        max_q_value = max(q_values.values())
        optimal_actions = [action for action, value in q_values.items() if value == max_q_value]
        return np.random.choice(optimal_actions)

    def update_q(self, state_idx, action_idx, reward, next_state_idx, next_action_idx):
        delta = reward + self.gamma * self.Q[next_state_idx, next_action_idx] - self.Q[state_idx, action_idx]
        self.E[state_idx, action_idx] += 1  
        for s in range(self.num_states):
            self.Q[s, :] += self.alpha * delta * self.E[s, :]
            self.E[s, :] *= self.gamma * self.lambda_ 
    
    def run(self, num_episodes=1000):
        learning_curve = []
        # for episode in range(num_episodes):
        while True:
            state = self.get_initial_state()
            state_idx = self.get_state_idx(state)
            action_idx = self.take_next_action(state_idx)
            total_reward = 0
            
            while state not in TERMINAL_STATES:
                next_state = self.get_next_state(state, ACTIONS[action_idx])
                next_state_idx = self.get_state_idx(next_state)
                reward = self.get_reward(state, next_state)
                total_reward += reward
                
                next_action_idx = self.take_next_action(next_state_idx)
                self.update_q(state_idx, action_idx, reward, next_state_idx, next_action_idx)
                state, action_idx = next_state, next_action_idx
            
            mse = estimate_mean_squared_error(self.V, OPTIMAL_POLICY_STATE_VALUES)
            learning_curve.append(mse)
            
            delta = max_norm(self.V, OPTIMAL_POLICY_STATE_VALUES)
            if delta < 1e-4:
                break

        policy = np.array([ACTIONS[np.argmax(self.Q[s])] for s in range(self.num_states)])
        return len(learning_curve), self.V, policy, learning_curve
    
def convert_state_to_row_col(state, num_cols):
    row = state // num_cols
    col = state % num_cols
    return row, col

def convert_row_col_idx_to_state_idx(row, col, num_cols):
    return row * num_cols + col

def pretty_print_policy(policy, num_rows, num_cols):
    symbol_map = np.vectorize(lambda x: ACTIONS_TO_SYMBOLS.get(x, " "))
    symbol_policy = symbol_map(policy).reshape(num_rows, num_cols)
    
    for goal_state in GOAL_STATES:
        row_idx, col_idx = goal_state
        symbol_policy[row_idx][col_idx] = "G"
    
    grid_string = "-" * (num_cols * 4 - 1) + "\n" 
    for row in symbol_policy:
        grid_string += " | ".join(row) + "\n"
        grid_string += "-" * (num_cols * 4 - 1) + "\n" 
        
    print("Policy")
    print(grid_string.strip()) 
    print("\n")

def pretty_print_value_function(value_function, num_rows, num_cols):
    value_function = value_function.reshape(num_rows, num_cols)
    format_values = np.vectorize(lambda x: f"{x:.4f}")
    formatted_values = format_values(value_function)
    
    grid_string = "-" * (num_cols * 9 - 1) + "\n"
    for row in formatted_values:
        grid_string += " | ".join(row) + "\n"
        grid_string += "-" * (num_cols * 9 - 1) + "\n"
        
    print("Value Function")
    print(grid_string.strip()) 
    print("\n")

def display_results(total_iterations, policy, value_function):
    print(f"Total Iterations: {total_iterations}")
    print(f"Mean Squared Error: {estimate_mean_squared_error(value_function, OPTIMAL_POLICY_STATE_VALUES):.4f}")
    # print(f"Max Norm error: {max_norm(value_function, OPTIMAL_POLICY_STATE_VALUES):.4f}")
    pretty_print_policy(policy, NUM_ROWS, NUM_COLS)
    pretty_print_value_function(value_function, NUM_ROWS, NUM_COLS)

def max_norm(v1, v2):
    return np.max(np.abs(v1 - v2))

def estimate_mean_squared_error(v1, v2):
    return np.mean((v1 - v2)**2)

def plot_graph(x, y, xlabel, ylabel, title, save_fp):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_fp, dpi=300)
    plt.close()
    
if __name__ == "__main__":
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    lambda_ = 0.9
    model = TrueOnlineSARSA_CatVsMonsters(gamma=0.925, alpha=0.1, theta=1e-4, n=5, num_rows=5, num_cols=5, epsilon=0.1)
    total_itereations, policy, value_function, learning_curve = model.run()
    plot_graph(x=[i for i in range(len(learning_curve))], y=learning_curve, xlabel="Number of Iterations", ylabel="MSE Error", title="Learning Curve", save_fp="./output/true-online-sarsa-cat-vs-monsters-learning-curve.png")
    display_results(total_itereations, policy, value_function)
