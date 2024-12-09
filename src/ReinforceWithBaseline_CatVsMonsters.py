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
OBSTACLES = [(2, 1), (2, 2), (2, 3), (3, 2)] # locations of where obstace is present. Cat cannot move into these locations
MONSTER_STATES = [(0, 3), (4, 1)] # locations of where monster is present.
NUM_ROWS, NUM_COLS, NUM_STATES = 5, 5, 25
OPTIMAL_POLICY = np.array(["AR", "AD", "AL", "AD", "AD", "AR", "AR", "AR", "AR", "AD", "AU", None, None, None, "AD", "AU", "AL", None, "AR", "AD", "AU", "AR", "AR", "AR", "AU"])
OPTIMAL_POLICY_STATE_VALUES = np.array([2.6638, 2.9969, 2.8117, 3.6671, 4.8497, 2.9713, 3.5101, 4.0819, 4.8497, 7.1648, 2.5936, 0, 0, 0, 8.4687, 2.0992, 1.0849, 0, 8.6097, 9.5269, 1.0849, 4.9465, 8.4687, 9.5269, 0])

class ReinforceWithBaseline_CatVsMonsters:
    
    def init(self, ):
        pass
    
    def run():
        pass
    
    def get_reward(self, curr_state, next_state):
        if curr_state in TERMINAL_STATES and next_state in TERMINAL_STATES:
            return 0
        elif curr_state not in GOAL_STATES and next_state in GOAL_STATES:
            return 10
        elif next_state in MONSTER_STATES:
            return -8
        return -0.05

    def get_initial_state(self):
        invalid_state_idxs = {convert_row_col_idx_to_state_idx(row, col, self.num_cols) for row, col in OBSTACLES}
        invalid_state_idxs.add(convert_row_col_idx_to_state_idx(4, 4, self.num_cols))
        valid_states = [state for state in range(25) if state not in invalid_state_idxs]
        state_idx = np.random.choice(valid_states)
        row, col = convert_state_to_row_col(state_idx, self.num_cols)
        return (row, col)

    def is_valid_state(self, state):
        return state not in OBSTACLES and state[0] >= 0 and state[0] < self.num_rows and state[1] >= 0 and state[1] < self.num_cols
    
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
    model = ReinforceWithBaseline_CatVsMonsters(gamma=0.925, alpha=0.1, theta=1e-4, n=5, num_rows=5, num_cols=5, epsilon=0.1)
    total_itereations, policy, value_function, learning_curve = model.run(num_iterations=1000)
    plot_graph(x=[i for i in range(len(learning_curve))], y=learning_curve, xlabel="Number of Iterations", ylabel="MSE Error", title="Learning Curve", save_fp="./output/reinforce-with-baseline-cat-vs-monsters-learning-curve.png")
    display_results(total_itereations, policy, value_function)