import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import GridWorld
from agent import Agent
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env   = GridWorld(size=GRID_SIZE)
agent = Agent(NUM_STATES, NUM_ACTIONS, device)

rewards_per_episode = np.zeros(EPISODES)
steps_per_episode   = np.zeros(EPISODES)

for episode in range(EPISODES):
    state = env.reset()
    env.generate_obstacles()  # randomize obstacles each episode
    total_reward = 0
    steps = 0

    while True:
        action     = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.memory.push(state, action, next_state, reward, done)
        agent.optimize()

        state        = next_state
        total_reward += reward
        steps        += 1

        # prevent infinite loop
        if done or steps >= 200:
            break

    rewards_per_episode[episode] = total_reward
    steps_per_episode[episode]   = steps

    print(f"Episode {episode + 1}/{EPISODES} | Steps: {steps} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

print("\nTraining complete!")
torch.save(agent.policy_net.state_dict(), "gridworld_dqn.pt")
print("Model saved to gridworld_dqn.pt")

# --- Plot ---
plt.figure(figsize=(12, 5))

# Reward plot
plt.subplot(121)
sum_rewards = np.zeros(EPISODES)
for x in range(EPISODES):
    sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
plt.plot(sum_rewards)
plt.title("Rewards (100-ep sum)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

# Epsilon decay plot
epsilon_vals = [max(EPSILON_END, 1.0 - i * EPSILON_DECAY) for i in range(EPISODES)]
plt.subplot(122)
plt.plot(epsilon_vals)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")

plt.tight_layout()
plt.savefig("gridworld_plot.png")
print("Plot saved to gridworld_plot.png")
