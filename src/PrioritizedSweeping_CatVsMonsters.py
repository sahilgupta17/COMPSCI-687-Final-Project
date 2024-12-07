import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


# # Constants
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

class PrioritizedSweeping_CatVsMonsters:
    
    def __init__(self):
        pass
    def run(self):
        pass




# class SARSA_CatVsMonsters:
    
#     def __init__(self, gamma=0.925, alpha=0.0075, theta=1e-4, epsilon=0.05):
#         self.policy = self.initialize_policy()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.theta = theta
#         self.epsilon = epsilon
#         self.num_rows = 5
#         self.num_cols = 5
#         self.num_states = 25
#         self.V = self.initialize_value_function(NUM_STATES)
#         self.q_hat = self.initialize_q_hat()
        
#     def initialize_q_hat(self):
#         q_hat = {}
#         for state in range(self.num_states):
#             for action in ACTIONS:
#                 q_hat[f"{state}-{action}"] = 0
#         return q_hat
    
#     def run(self):
#         total_iterations = 0
#         total_actions = 0
#         learning_curve_actions_iterations = []
#         learning_curve_value_function = []
#         while True:
#             total_iterations += 1
#             episode = self.generate_episode()
#             V_copy = self.V.copy()
#             for idx in range(len(episode)):
#                 total_actions += 1
#                 learning_curve_actions_iterations.append((total_actions, total_iterations))
#                 state_idx, action, reward = episode[idx]
#                 if idx == len(episode) - 1:
#                      self.q_hat[f"{state_idx}-{action}"] = self.q_hat[f"{state_idx}-{action}"] + self.alpha * (reward - self.q_hat[f"{state_idx}-{action}"])
#                 else:
#                     self.q_hat[f"{state_idx}-{action}"] = self.q_hat[f"{state_idx}-{action}"] + self.alpha * (reward + (self.gamma * self.q_hat[f"{episode[idx+1][0]}-{episode[idx+1][1]}"]) - self.q_hat[f"{state_idx}-{action}"])
        
            
#             for state_idx in range(NUM_STATES):
#                 state = convert_state_to_row_col(state_idx, self.num_cols)
#                 if state not in OBSTACLES:
#                     actions, probabilities = self.get_all_action_probabilities(state_idx)
#                     V_updated = 0
#                     for action, probability in zip(actions, probabilities):
#                         V_updated += (probability * self.q_hat[f"{state_idx}-{action}"])
#                     self.V[state_idx] = V_updated
#                     self.policy[state_idx] = ACTIONS[np.argmax([self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS])]
            
#             delta = max_norm(V_copy, self.V)
#             learning_curve_value_function.append((total_iterations, mean_squared(self.V, OPTIMAL_POLICY_STATE_VALUES)))
#             if total_iterations % 10000 == 0:
#                 print(f"Iteration: {total_iterations}")
            
#             if delta < self.theta:
#                 break
                    
#         return total_iterations, self.policy, self.V, learning_curve_actions_iterations, learning_curve_value_function
    
#     def get_all_action_probabilities(self, state_idx):
#         q_values = {action: self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS}
        
#         max_q_value = max(q_values.values())
#         optimal_actions = [action for action, value in q_values.items() if value == max_q_value]
        
#         action_probabilities = {}
#         num_actions = len(ACTIONS)
#         num_optimal_actions = len(optimal_actions)
        
#         for action in ACTIONS:
#             if action in optimal_actions:
#                 action_probabilities[action] = ((1 - self.epsilon) / num_optimal_actions) + (self.epsilon / num_actions)
#             else:
#                 action_probabilities[action] = self.epsilon / num_actions
        
#         actions, probabilities = zip(*action_probabilities.items())
#         return actions, probabilities
        
#     def generate_episode(self):
#         episode = []
#         state = self.get_initial_state()
#         while state not in TERMINAL_STATES:
#             state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
#             action = self.take_next_action(state)
#             next_state = self.get_next_state(state, action)
#             reward = self.get_reward(state, next_state)
#             episode.append((state_idx, action, reward))
#             state = next_state
#         return episode
    
#     def take_next_action(self, state):
#         state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
        
#         q_values = {action: self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS}
        
#         max_q_value = max(q_values.values())
#         optimal_actions = [action for action, value in q_values.items() if value == max_q_value]
        
#         action_probabilities = {}
#         num_actions = len(ACTIONS)
#         num_optimal_actions = len(optimal_actions)
        
#         for action in ACTIONS:
#             if action in optimal_actions:
#                 action_probabilities[action] = ((1 - self.epsilon) / num_optimal_actions) + (self.epsilon / num_actions)
#             else:
#                 action_probabilities[action] = self.epsilon / num_actions
        
#         actions, probabilities = zip(*action_probabilities.items())
        
#         # print(self.epsilon, actions, probabilities)
#         chosen_action = np.random.choice(actions, p=probabilities)
    
#         return chosen_action
    
#     def get_next_state(self, curr_state, action):
        
#         if curr_state in TERMINAL_STATES:
#             return curr_state
        
#         random_num = np.random.rand()
#         next_state = None
#         actual_movement_direction = None
        
#         movement_directions = {
#             "AU": (-1, 0),  # Up
#             "AD": (1, 0),   # Down
#             "AL": (0, -1),  # Left
#             "AR": (0, 1)    # Right
#         }
        
#         if random_num <= 0.70:
#             # cat moves in desired direction
#             actual_movement_direction = action
#         elif 0.7 < random_num <= 0.82:
#             # cat moves 90 degrees to the right action
#             actual_movement_direction = ACTIONS[(ACTIONS.index(action) + 1) % 4]
#         elif 0.82 < random_num <= 0.94:
#             # cat moves 90 degrees to the left action
#             actual_movement_direction = ACTIONS[(ACTIONS.index(action) - 1) % 4]
#         else:
#             return curr_state
        
#         next_state = (curr_state[0] + movement_directions[actual_movement_direction][0], curr_state[1] + movement_directions[actual_movement_direction][1])
        
#         return next_state if self.is_valid_state(next_state) else curr_state
    
#     def get_reward(self, curr_state, next_state):
#         if curr_state in TERMINAL_STATES and next_state in TERMINAL_STATES:
#             return 0
#         elif curr_state not in GOAL_STATES and next_state in GOAL_STATES:
#             return 10
#         elif next_state in MONSTER_STATES:
#             return -8
#         return -0.05
    
#     def get_initial_state(self):    
#         invalid_state_idxs = {convert_row_col_idx_to_state_idx(row, col, self.num_cols) for row, col in OBSTACLES}
#         invalid_state_idxs.add(convert_row_col_idx_to_state_idx(4, 4, self.num_cols))
#         valid_states = [state for state in range(25) if state not in invalid_state_idxs]
#         state_idx = np.random.choice(valid_states)
#         row, col = convert_state_to_row_col(state_idx, self.num_cols)
#         return (row, col)

#     def is_valid_state(self, state):
#         return state not in OBSTACLES and state[0] >= 0 and state[0] < self.num_rows and state[1] >= 0 and state[1] < self.num_cols
    
#     def initialize_policy(self):
#         policy = np.random.choice(ACTIONS, NUM_ROWS * NUM_COLS)
#         for obstacle_state in OBSTACLES:
#             row_idx, col_idx = obstacle_state
#             policy[convert_row_col_idx_to_state_idx(row_idx, col_idx, NUM_COLS)] = None
#         return policy
                    
#     def initialize_value_function(self, num_states):
#         # return np.zeros(num_states)
#         return np.random.rand(num_states)
    
# def convert_state_to_row_col(state, num_cols):
#     row = state // num_cols
#     col = state % num_cols
#     return row, col

# def convert_row_col_idx_to_state_idx(row, col, num_cols):
#     return row * num_cols + col

# def pretty_print_policy(policy, num_rows, num_cols):
#     symbol_map = np.vectorize(lambda x: ACTIONS_TO_SYMBOLS.get(x, " "))
#     symbol_policy = symbol_map(policy).reshape(num_rows, num_cols)
    
#     for goal_state in GOAL_STATES:
#         row_idx, col_idx = goal_state
#         symbol_policy[row_idx][col_idx] = "G"
    
#     grid_string = "-" * (num_cols * 4 - 1) + "\n" 
#     for row in symbol_policy:
#         grid_string += " | ".join(row) + "\n"
#         grid_string += "-" * (num_cols * 4 - 1) + "\n" 
        
#     print("Policy")
#     print(grid_string.strip()) 
#     print("\n")

# def pretty_print_value_function(value_function, num_rows, num_cols):
#     value_function = value_function.reshape(num_rows, num_cols)
#     format_values = np.vectorize(lambda x: f"{x:.4f}")
#     formatted_values = format_values(value_function)
    
#     grid_string = "-" * (num_cols * 9 - 1) + "\n"
#     for row in formatted_values:
#         grid_string += " | ".join(row) + "\n"
#         grid_string += "-" * (num_cols * 9 - 1) + "\n"
        
#     print("Value Function")
#     print(grid_string.strip()) 
#     print("\n")

# def display_results(total_iterations, policy, value_function):
#     print(f"Total Iterations: {total_iterations} \n")
#     pretty_print_policy(policy, NUM_ROWS, NUM_COLS)
#     pretty_print_value_function(value_function, NUM_ROWS, NUM_COLS)
    
# def max_norm(v1, v2):
#     return np.max(np.abs(v1 - v2))

# def plot_graph(x, y, xlabel, ylabel, title, save_fp):
#     plt.plot(x, y)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.savefig(save_fp, dpi=300)
#     plt.close()
    
# def mean_squared(v1, v2):
#     return np.mean((v1 - v2) ** 2)

# def generate_x_y(learning_curve_all_runs, total_runs):
#     # Find the maximum length of runs
#     max_len = max(len(run) for run in learning_curve_all_runs)

#     # Initialize arrays with NaN to handle varying run lengths
#     x = np.full((total_runs, max_len), fill_value=np.nan)
#     y = np.full((total_runs, max_len), fill_value=np.nan)

#     # Populate x and y arrays
#     for idx, learning_curve in enumerate(learning_curve_all_runs):
#         x_run = [step[0] for step in learning_curve]
#         y_run = [step[1] for step in learning_curve]
        
#         # Add values to x and y arrays
#         for i in range(len(x_run)):
#             x[idx, i] = x_run[i]
#             y[idx, i] = y_run[i]

#     # Compute averages ignoring NaNs
#     x = np.nanmean(x, axis=0)
#     y = np.nanmean(y, axis=0)
#     return x, y

# # if __name__ == '__main__':
    
# #     # testing
# #     # sarsa_model = SARSA_CatVsMonsters(gamma=0.925, alpha=0.05, theta=1e-4, epsilon=0.05)
# #     # iteration, policy, value_function, learning_curve_actions_iterations = sarsa_model.run()
# #     # display_results(iteration, policy, value_function)
# #     # max_norm_error = max_norm(value_function, OPTIMAL_POLICY_STATE_VALUES)
# #     # mean_squared_error = mean_squared(value_function, OPTIMAL_POLICY_STATE_VALUES)
# #     # print(f"Max Norm Error: {max_norm_error}")
# #     # print(f"Mean Squared Error: {mean_squared_error}")
# #     # x = [step[0] for step in learning_curve_actions_iterations]
# #     # y = [step[1] for step in learning_curve_actions_iterations]
# #     # plot_graph(x, y, "Total Actions Taken", "Number of episodes completed", "Learning curve", "Q2b-testing.png")
    
# #     # 2b
# #     learning_curve_all_runs = []
# #     value_function_all_runs = []
# #     learning_curve_value_function_all_runs = []
# #     policy_all_runs = []
# #     total_runs = 20

# #     for run in range(total_runs):
# #         print(f"Run: {run}")
# #         sarsa_model = SARSA_CatVsMonsters(gamma=0.925, alpha=0.1, theta=1e-4, epsilon=0.05)
# #         iteration, policy, value_function, learning_curve_actions_iterations, learning_curve_value_function = sarsa_model.run()
# #         learning_curve_all_runs.append(learning_curve_actions_iterations)
# #         learning_curve_value_function_all_runs.append(learning_curve_value_function)
# #         value_function_all_runs.append(value_function)
# #         policy_all_runs.append(policy)

# #     x, y = generate_x_y(learning_curve_all_runs, total_runs)
# #     average_value_function = np.mean(value_function_all_runs, axis=0)

# #     max_policy_length = max(len(policy) for policy in policy_all_runs)
# #     policy_modes = []

# #     for col in range(max_policy_length):
# #         column_values = [policy[col] for policy in policy_all_runs if col < len(policy)]

# #         frequency = {}
# #         for value in column_values:
# #             frequency[value] = frequency.get(value, 0) + 1

# #         most_common = max(frequency, key=frequency.get)
# #         policy_modes.append(most_common)
        
# #     average_policy = policy_modes

# #     plot_graph(x, y, "Total Actions Taken", "Number of episodes completed", "Learning curve", "q2b-learning-curve.png")
# #     max_norm_error = max_norm(average_value_function, OPTIMAL_POLICY_STATE_VALUES)
# #     mean_squared_error = mean_squared(average_value_function, OPTIMAL_POLICY_STATE_VALUES)
# #     print(f"Max Norm Error: {max_norm_error}")
# #     print(f"Mean Squared Error: {mean_squared_error}")

# #     # 2c
# #     x, y = generate_x_y(learning_curve_value_function_all_runs, total_runs)
# #     plot_graph(x, y, "Number of episodes", "Mean Squared Error", "Learning curve", "q2c-learning-curve-value-function.png")

# #     # 2d
# #     display_results(iteration, average_policy, average_value_function)



    
    

    
    

