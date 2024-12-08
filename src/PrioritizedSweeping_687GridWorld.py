import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import heapq

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
OBSTACLES = [(2, 2), (3, 2)] # locations of where obstace is present. Cat cannot move into these locations
WATER_STATE = [(4, 2)] # locations of where monster is present.
NUM_ROWS, NUM_COLS, NUM_STATES = 5, 5, 25

class PrioritizedSweeping_687GridWorld:
    
    def __init__(self, gamma, alpha, theta, n, num_rows, num_cols, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.theta = theta
        self.n = n
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states = self.num_rows * self.num_cols
        self.epsilon = epsilon
        self.policy = self.initialize_policy()
        self.q_hat = self.initialize_q_hat()
        self.model = {} 
        self.V = self.initialize_value_function(self.num_states)
        
    def initialize_value_function(self, num_states):
        return np.zeros(num_states)
    
    def initialize_policy(self):
        policy = np.random.choice(ACTIONS, NUM_ROWS * NUM_COLS)
        for obstacle_state in OBSTACLES:
            row_idx, col_idx = obstacle_state
            policy[convert_row_col_idx_to_state_idx(row_idx, col_idx, NUM_COLS)] = None
        return policy
    
    def initialize_q_hat(self):
        q_hat = {}
        for state in range(self.num_states):
            for action in ACTIONS:
                q_hat[f"{state}-{action}"] = 0
        return q_hat

    def update_model(self, state_idx, action, reward, next_state_idx):
        self.model[f"{state_idx}-{action}"] = (reward, next_state_idx)
        
    
    def get_predecessors(self, state_idx):
        predecessors = []
        for action in ACTIONS:
            for s_idx in range(self.num_states):
                if f"{s_idx}-{action}" in self.model:
                    _, next_s_idx = self.model[f"{s_idx}-{action}"]
                    if next_s_idx == state_idx:
                        predecessors.append((s_idx, action))
        return predecessors
    
    def get_all_predecessors(self):
        predecessors = {}
        for state_idx in range(NUM_STATES):
            for action in ACTIONS:
                next_state = self.get_next_state(convert_state_to_row_col(state_idx, NUM_COLS), action)
                next_state_idx = convert_row_col_idx_to_state_idx(next_state[0], next_state[1], NUM_COLS)
                if next_state_idx not in predecessors:
                    predecessors[next_state_idx] = []
                predecessors[next_state_idx].append((state_idx, action))
        return predecessors
    
    def run(self):
        total_iterations = 0
        priority_queue = []

        while True:
            total_iterations += 1
            state = self.get_initial_state()
            while state not in TERMINAL_STATES:
                state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
                action = self.take_next_action(state_idx)
                next_state = self.get_next_state(state, action)
                next_state_idx = convert_row_col_idx_to_state_idx(next_state[0], next_state[1], self.num_cols)
                reward = self.get_reward(state, next_state)

                # Update the model
                self.model[f"{state_idx}-{action}"] = (reward, next_state_idx)

                # Compute priority
                max_next_q = max(self.q_hat[f"{next_state_idx}-{a}"] for a in ACTIONS)
                priority = abs(reward + (self.gamma * max_next_q) - self.q_hat[f"{state_idx}-{action}"])

                if priority > self.theta:
                    heapq.heappush(priority_queue, (-priority, state_idx, action))

                # Update Q-values using the priority queue
                for _ in range(self.n):
                    if not priority_queue:
                        break
                    priority, state_idx, action = heapq.heappop(priority_queue)
                    reward, next_state_idx = self.model[f"{state_idx}-{action}"]
                    max_nex_state_q_value = max(self.q_hat[f"{next_state_idx}-{action_from_next_state}"] for action_from_next_state in ACTIONS)
                    self.q_hat[f"{state_idx}-{action}"] = self.q_hat[f"{state_idx}-{action}"] + (self.alpha * (reward + (self.gamma * max_nex_state_q_value) - self.q_hat[f"{state_idx}-{action}"]))

                    # Update priorities for predecessors
                    for pred_state_idx, pred_action in self.get_predecessors(state_idx):
                        pred_reward, pred_next_state = self.model[f"{pred_state_idx}-{pred_action}"]
                        pred_max_q = max(self.q_hat[f"{pred_next_state}-{a_prime}"] for a_prime in ACTIONS)
                        pred_priority = abs(
                            pred_reward + (self.gamma * pred_max_q) - self.q_hat[f"{pred_state_idx}-{pred_action}"]
                        )
                        if pred_priority > self.theta:
                            heapq.heappush(priority_queue, (-pred_priority, pred_state_idx, pred_action))

                # Update the state and policy
                self.update_policy_and_value_function()
                state = next_state
                
            if total_iterations % 500 == 0:
                print(f"Total Iterations: {total_iterations}")
                pretty_print_value_function(self.V, self.num_rows, self.num_cols)

            if not priority_queue:
                break

        return total_iterations, self.policy, self.V
    
    def update_policy_and_value_function(self):
        for state_idx in range(self.num_states):
            state = convert_state_to_row_col(state_idx, self.num_cols)
            if state not in OBSTACLES and state not in TERMINAL_STATES:
                q_values = [self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS]
                self.policy[state_idx] = ACTIONS[np.argmax(q_values)]
                self.V[state_idx] = max(q_values)
                
    def take_next_action(self, state_idx):
        # state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
        q_values = {action: self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS}
        max_q_value = max(q_values.values())
        optimal_actions = [action for action, value in q_values.items() if value == max_q_value]
        return np.random.choice(optimal_actions)
    
    # def take_next_action(self, state_idx):
    #     # state_idx = convert_row_col_idx_to_state_idx(state[0], state[1], self.num_cols)
        
    #     q_values = {action: self.q_hat[f"{state_idx}-{action}"] for action in ACTIONS}
        
    #     max_q_value = max(q_values.values())
    #     optimal_actions = [action for action, value in q_values.items() if value == max_q_value]
        
    #     action_probabilities = {}
    #     num_actions = len(ACTIONS)
    #     num_optimal_actions = len(optimal_actions)
        
    #     for action in ACTIONS:
    #         if action in optimal_actions:
    #             action_probabilities[action] = ((1 - self.epsilon) / num_optimal_actions) + (self.epsilon / num_actions)
    #         else:
    #             action_probabilities[action] = self.epsilon / num_actions
        
    #     actions, probabilities = zip(*action_probabilities.items())
    #     chosen_action = np.random.choice(actions, p=probabilities)
    #     return chosen_action
    
    def get_next_state(self, curr_state, action):
        
        if curr_state in TERMINAL_STATES:
            return curr_state
        
        random_num = np.random.rand()
        next_state = None
        actual_movement_direction = None
        
        movement_directions = {
            "AU": (-1, 0),  # Up
            "AD": (1, 0),   # Down
            "AL": (0, -1),  # Left
            "AR": (0, 1)    # Right
        }
        
        if random_num <= 0.80:
            # cat moves in desired direction
            actual_movement_direction = action
        elif 0.80 < random_num <= 0.85:
            # cat moves 90 degrees to the right action
            actual_movement_direction = ACTIONS[(ACTIONS.index(action) + 1) % 4]
        elif 0.85 < random_num <= 0.90:
            # cat moves 90 degrees to the left action
            actual_movement_direction = ACTIONS[(ACTIONS.index(action) - 1) % 4]
        else:
            return curr_state
        
        next_state = (curr_state[0] + movement_directions[actual_movement_direction][0], curr_state[1] + movement_directions[actual_movement_direction][1])
        return next_state if self.is_valid_state(next_state) else curr_state

    def get_reward(self, curr_state, next_state):
        if curr_state in TERMINAL_STATES and next_state in TERMINAL_STATES:
            return 0
        elif curr_state not in GOAL_STATES and next_state in GOAL_STATES:
            return 10
        elif next_state in WATER_STATE:
            return -10
        return 0

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
    print(f"Total Iterations: {total_iterations} \n")
    # print(f"Mean Squared Error: {mean_squared(value_function, OPTIMAL_POLICY_STATE_VALUES)} \n")
    # print(f"Max Norm error: {max_norm(value_function, OPTIMAL_POLICY_STATE_VALUES)} \n")
    pretty_print_policy(policy, NUM_ROWS, NUM_COLS)
    pretty_print_value_function(value_function, NUM_ROWS, NUM_COLS)

def max_norm(v1, v2):
    return np.max(np.abs(v1 - v2))

def plot_graph(x, y, xlabel, ylabel, title, save_fp):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_fp, dpi=300)
    plt.close()
    
def mean_squared(v1, v2):
    return np.mean((v1 - v2) ** 2)

if __name__ == "__main__":
    model = PrioritizedSweeping_687GridWorld(gamma=0.925, alpha=0.05, theta=1e-2, n=5, num_rows=5, num_cols=5, epsilon=0.1)
    total_itereations, policy, value_function = model.run()
    display_results(total_itereations, policy, value_function)
    