'''
Model made to solve moving through 5x5 gridworld environment with pitfalls and goal
i.e. 25 states (rows) and 4 actions (columns) (UP, DOWN, LEFT, RIGHT)
Reward matrix defined
Epsilon-greedy policy used for action selection
'''
# Q-table initialization
import numpy as np
grid_size = 5
n_actions = 4
# Initialize Q-table with zeros
Q_table = np.zeros((grid_size * grid_size, n_actions))

# Reward matrix definition
rewards = np.full((grid_size * grid_size,), -1)
rewards[24] = 10  # Goal state
rewards[12] = -10  # Pitfall state

# Epsilon-greedy action selection
def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit
    
# Q-learning update rule
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(1000):
    state = np.random.randint(0, grid_size * grid_size)  # Random start
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, grid_size * grid_size)  # Random next state
        reward = rewards[next_state]

        # Update Q-value using Bellman equation
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])

        state = next_state
        if next_state == 24 or next_state == 12:
            done = True  # End episode if goal or pitfall is reached

'''
Same problem but using policy gradients solution
'''

# Build poilicy network
import tensorflow as tf

n_states = grid_size * grid_size  # 25 states in the grid
n_actions = 4  # Four possible actions

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')  # Output action probabilities
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Action selection using policy network
def get_action(state):
    state_input = tf.one_hot(state, n_states)  # One-hot encode the state
    action_probs = model(state_input[np.newaxis, :])
    return np.random.choice(n_actions, p=action_probs.numpy()[0])

# Cumulative reward calculation
def compute_cumulative_rewards(rewards, gamma=0.99):
    cumulative_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        cumulative_rewards[t] = running_add
    return cumulative_rewards

# update policy using REINFORCE algorithm
def update_policy(states, actions, rewards):
    cumulative_rewards = compute_cumulative_rewards(rewards)

    with tf.GradientTape() as tape:
        state_inputs = tf.one_hot(states, n_states)
        action_probs = model(state_inputs)
        action_masks = tf.one_hot(actions, n_actions)

        # Log-probabilities of the actions taken
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)

        # Policy loss function
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)

    # Apply gradients to update the policy network
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


'''
Evaluation metrics for reinforcement learning models
Cumulative reward, Episode length, Success rate, and exploration vs exploitation ratio
'''

# Cumulative Reward
import matplotlib.pyplot as plt
# Calculate and store cumulative rewards
cumulative_rewards = []
for episode in range(1000):
    total_reward = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    cumulative_rewards.append(total_reward)
# Plot the cumulative rewards over episodes
plt.plot(cumulative_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Episodes')
plt.show()

# Episode Length
episode_lengths = []
actions = []
for episode in range(1000):
    steps = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        
        next_state = np.random.randint(0, n_states)
        steps += 1
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    episode_lengths.append(steps)
# Plot a histogram of episode lengths
plt.hist(episode_lengths, bins=20)
plt.xlabel('Episode Length (Steps)')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Lengths')
plt.show()

# Success Rate and Exploration vs Exploitation Ratio
# Redefine epsilon_greedy_action to log explorations & exploitations
actions = []
def epsilon_greedy_action(Q_table, state, epsilon):
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        actions.append('explore')
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        actions.append('exploit')
        return np.argmax(Q_table[state])  # Action with the highest Q-value
# Calculate and store cumulative rewards and actions
cumulative_rewards = []
for episode in range(1000):
    total_reward = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    cumulative_rewards.append(total_reward)
# Calculate success rate
success_count = sum(1 for reward in cumulative_rewards if reward >= 10)
success_rate = success_count / len(cumulative_rewards)
# Exploration vs. exploitation ratio
#print(actions)
exploration_count = sum(1 for action in actions if action == 'explore')
exploitation_count = sum(1 for action in actions if action == 'exploit')
exploration_exploitation_ratio = exploration_count / (exploration_count + exploitation_count)
print(f"Success Rate: {success_rate * 100}%")
print(f"Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")