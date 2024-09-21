# Preprocessing Script

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('your_data.csv')

# Preprocess the data
# Convert Date to numerical value (e.g., ordinal representation)
data['Date'] = pd.to_datetime(data['Date'])
data['Date_Ordinal'] = data['Date'].apply(lambda x: x.toordinal())

# Encode Region_Name using LabelEncoder for numerical processing
label_encoder = LabelEncoder()
data['Region_Code'] = label_encoder.fit_transform(data['Region_Name'])

# Normalize the Average_Price_SA for better RL learning performance.
data['Normalized_Average_Price'] = (data['Average_Price_SA'] - data['Average_Price_SA'].mean()) / data['Average_Price_SA'].std()

# The state space will consist of Date_Ordinal and Region_Code
states = data[['Date_Ordinal', 'Region_Code']].values
prices = data['Normalized_Average_Price'].values

# Define the action space (price adjustments)
action_space = [-0.01, 0, 0.01]  # Price decrease, maintain, or increase


# Q-Learning Script with Graphs

import numpy as np
import matplotlib.pyplot as plt

# Q-learning algorithm for dynamic pricing

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration factor (we will decay this over time)
epsilon_decay = 0.99
min_epsilon = 0.01
episodes = 1000  # Number of training episodes

# Initialize Q-table
q_table = np.zeros((len(states), len(action_space)))

# Rewards will be based on price changes that increase revenue (simulated reward)
def calculate_reward(price_before, price_after):
    reward = price_after - price_before  # Simple reward based on price change
    return reward

# Tracking rewards and epsilon
rewards_per_episode = []
epsilon_values = []

# Q-learning training loop
for episode in range(episodes):
    state_index = np.random.randint(0, len(states))  # Randomly initialize a state
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, len(action_space))  # Explore
        else:
            action_index = np.argmax(q_table[state_index])  # Exploit learned values

        # Take the action (price adjustment)
        current_price = prices[state_index]
        price_change = action_space[action_index]
        new_price = current_price + price_change

        # Calculate reward
        reward = calculate_reward(current_price, new_price)
        total_reward += reward

        # Update Q-table
        q_table[state_index, action_index] = q_table[state_index, action_index] + alpha * (
            reward + gamma * np.max(q_table[state_index]) - q_table[state_index, action_index]
        )

        done = True  # End the episode after one action

    # Track total rewards and epsilon decay per episode
    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)

    # Decay epsilon after each episode
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

# Plotting epsilon decay and rewards over time
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Epsilon decay
ax[0].plot(epsilon_values, label='Epsilon', color='blue')
ax[0].set_title('Epsilon Decay Over Episodes')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Epsilon')
ax[0].grid(True)

# Rewards over episodes
ax[1].plot(rewards_per_episode, label='Reward', color='green')
ax[1].set_title('Rewards Over Episodes')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total Reward')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Track the actions taken during the training
actions_taken = []

for episode in range(episodes):
    state_index = np.random.randint(0, len(states))  # Randomly initialize a state
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, len(action_space))  # Explore
        else:
            action_index = np.argmax(q_table[state_index])  # Exploit learned values

        # Track the action taken
        actions_taken.append(action_space[action_index])

        done = True  # End the episode

# Convert the actions into a numpy array for easier plotting
actions_taken = np.array(actions_taken)

# Plot the frequency of actions taken (price adjustments)
plt.figure(figsize=(10, 6))
plt.hist(actions_taken, bins=3, edgecolor='black', color='orange')
plt.title('Frequency of Actions Taken During Q-Learning')
plt.xlabel('Action (Price Adjustment: -0.01 = Decrease, 0 = Maintain, 0.01 = Increase)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
