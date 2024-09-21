
# Dynamic Pricing with Q-Learning

## Overview

This project implements a dynamic pricing strategy using the **Q-Learning** algorithm, a basic reinforcement learning method. The goal is to adjust product prices dynamically based on historical sales data, which consists of prices for different regions over time. The model is designed to maximize revenue by learning optimal price adjustments.

### Files

- `q_learning_dynamic_pricing_scripts.txt`: Contains the preprocessing code and Q-learning implementation with visualizations.
- `your_data.csv`: The dataset used for dynamic pricing analysis, containing average prices, regions, and dates.

## Project Components

1. **Preprocessing**:
   - The raw data is processed to convert dates into ordinal form, encode regions as numeric values, and normalize average prices.
   - The preprocessed data is used as input for the Q-Learning model, where each state consists of a date and region, and the action space represents price adjustments (-1%, 0%, +1%).

2. **Q-Learning Model**:
   - A Q-learning agent is trained to dynamically adjust prices by exploring the state-action space. The rewards are based on simulated price changes, and the model updates its Q-values using the Bellman equation.
   - The model starts by exploring the pricing actions using an epsilon-greedy policy, and gradually shifts towards exploiting the learned pricing strategies as epsilon decays.

3. **Visualization**:
   - **Epsilon Decay**: Shows how the model transitions from exploration to exploitation over time.
   - **Rewards Over Episodes**: Tracks how the model's performance (in terms of reward) improves over the course of training.
   - **Actions Taken**: Displays the frequency of pricing adjustments (increase, decrease, or maintain) during the learning process.

## How to Use

1. **Preprocessing**:
   - Ensure that the dataset (`your_data.csv`) is in the correct format with columns: `Date`, `Region_Name`, `Average_Price_SA`, etc.
   - Run the preprocessing script to convert the data into a suitable format for the Q-learning model.

2. **Q-Learning**:
   - Use the Q-learning script to train the agent over a series of episodes. The agent will learn the optimal pricing strategies for different regions and dates.
   - Visualize the model's learning process using the provided graphs for epsilon decay, rewards, and actions.

3. **Results**:
   - The output graphs will provide insights into the model's performance and decision-making process.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sklearn`

## Future Improvements

- Incorporate a more complex reward function based on actual revenue and sales data.
- Expand the action space to include more granular price adjustments.
- Implement more advanced RL algorithms such as **Deep Q-Networks (DQN)** or **Proximal Policy Optimization (PPO)** for handling larger and more complex state-action spaces.

## License

This project is open-source and available under the MIT License.
