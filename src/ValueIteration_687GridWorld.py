import numpy as np # type: ignore
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
TERMINAL_STATES = [(4, 4)] # only state 21 is terminal. Optionally state with catnip is also terminal when explicitly enabled
OBSTACLES = [(2, 2), (3, 2)] # locations of where obstace is present. Cat cannot move into these locations
WATER_STATE = [(4, 2)] # locations of where monster is present.
NUM_ROWS, NUM_COLS, NUM_STATES = 5, 5, 25

class ValueIteration:
    def __init__(self, gamma=0.925, num_rows=5, num_cols=5, theta=1e-4,useStandardVersion=True):
        self.gamma = gamma
        self.theta = theta
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = num_rows * num_cols
        self.useStandardVersion = useStandardVersion
        self.V = initialize_value_function(self.num_states)

    def value_iteration(self):
        iterations = 0
        all_states = [(i, j) for i in range(self.num_rows) for j in range(self.num_cols)]
        while True:
            delta = 0
            V_next = np.zeros(self.num_states)
            V_copy = np.copy(self.V)
            for state in all_states:
                if state in OBSTACLES:
                    continue
                state_idx = convert_row_col_idx_to_state_idx(row=state[0], col=state[1], num_cols=self.num_cols)
                max_value = float("-inf")
                for action in ACTIONS:
                    value = 0
                    all_possible_next_states_with_transition_probabilities = self.get_all_possible_next_states_with_transition_probabilities(state, action)
                    for next_state, transition_probability in all_possible_next_states_with_transition_probabilities:
                        next_state_idx = convert_row_col_idx_to_state_idx(row=next_state[0], col=next_state[1], num_cols=self.num_cols)
                        reward = self.get_reward(curr_state=state, next_state=next_state)
                        value += (transition_probability * (reward + (self.gamma * self.V[next_state_idx])))
                    max_value = max(max_value, value)   
                V_next[state_idx] = max_value
                if not self.useStandardVersion:
                    self.V[state_idx] = max_value
                delta = max(delta, abs(V_copy[state_idx] - V_next[state_idx]))
                
            iterations += 1
            self.V = V_next
            
            if delta < self.theta:
                break
            
        self.policy = np.full(25, None)
        for state in all_states:
            if state in OBSTACLES:
                continue
            state_idx = convert_row_col_idx_to_state_idx(row=state[0], col=state[1], num_cols=self.num_cols)
            max_value = float("-inf")
            max_value_action = None
            for action in ACTIONS:
                value = 0
                all_possible_next_states_with_transition_probabilities = self.get_all_possible_next_states_with_transition_probabilities(state, action)
                for next_state, transition_probability in all_possible_next_states_with_transition_probabilities:
                    next_state_idx = convert_row_col_idx_to_state_idx(row=next_state[0], col=next_state[1], num_cols=self.num_cols)
                    reward = self.get_reward(curr_state=state, next_state=next_state)
                    value += (transition_probability * (reward + (self.gamma * self.V[next_state_idx])))
                if value > max_value:
                    max_value_action = action
                    max_value = value
            self.policy[state_idx] = max_value_action
        return self.policy, self.V, iterations

    def run(self):
        return self.value_iteration()
    
    def get_reward(self, curr_state, next_state):
        if curr_state in TERMINAL_STATES and next_state in TERMINAL_STATES:
            return 0
        elif curr_state not in GOAL_STATES and next_state in GOAL_STATES:
            return 10
        elif next_state in WATER_STATE:
            return -10
        return 0
    
    def get_all_possible_next_states_with_transition_probabilities(self, curr_state, action):
        if curr_state in TERMINAL_STATES:
            return [(curr_state, 1.0)]
        
        movement_directions = {
            "AU": (-1, 0),  # Up
            "AD": (1, 0),   # Down
            "AL": (0, -1),  # Left
            "AR": (0, 1)    # Right
        }
        
        next_states = []
        action_idx = ACTIONS.index(action)
        # cat gets lazy and doesn't move
        next_states.append((curr_state, 0.1))
        
        # cat moves in intended direction
        next_state = (curr_state[0] + movement_directions[action][0], curr_state[1] + movement_directions[action][1])
        next_state = next_state if self.is_valid_state(next_state) else curr_state
        next_states.append((next_state, 0.8))
        
        # cat gets confused and moves in a direction that is 90 degrees to the right
        next_state = (curr_state[0] + movement_directions[ACTIONS[(action_idx + 1) % len(ACTIONS)]][0], curr_state[1] + movement_directions[ACTIONS[(action_idx + 1) % len(ACTIONS)]][1])
        next_state = next_state if self.is_valid_state(next_state) else curr_state
        next_states.append((next_state, 0.05))
        
        # cat gets confused and moves in a direction that is 90 degrees to the left
        next_state = (curr_state[0] + movement_directions[ACTIONS[(action_idx - 1) % len(ACTIONS)]][0], curr_state[1] + movement_directions[ACTIONS[(action_idx - 1) % len(ACTIONS)]][1])
        next_state = next_state if self.is_valid_state(next_state) else curr_state
        next_states.append((next_state, 0.05))

        return next_states
        
    def is_valid_state(self, state):
        return state not in OBSTACLES and state[0] >= 0 and state[0] < self.num_rows and state[1] >= 0 and state[1] < self.num_cols

def initialize_value_function(num_states):
    return np.zeros(num_states)

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
    print(f"Total Iterations: {total_iterations} \n")
    pretty_print_policy(policy, num_rows, num_cols)
    pretty_print_value_function(value_function, num_rows, num_cols)
    
if __name__ == '__main__':
    
    # Environment parameters
    num_rows, num_cols = 5, 5
    num_states = num_rows * num_cols
    
    # Q1
    # standard version
    gamma = 0.925
    vi_algo = ValueIteration(gamma=gamma, num_rows=num_rows, num_cols=num_cols, useStandardVersion=True)
    optimal_policy, optimal_value_function, total_iterations = vi_algo.run()
    print("Standard version")
    display_results(total_iterations, optimal_policy, optimal_value_function)
    
    # # in-place version
    # gamma = 0.925
    # vi_algo = ValueIteration(gamma=gamma, num_rows=num_rows, num_cols=num_cols, useStandardVersion=False)
    # optimal_policy, optimal_value_function, total_iterations = vi_algo.run()
    # print("In-place version")
    # display_results(total_iterations, optimal_policy, optimal_value_function)
    
    
    