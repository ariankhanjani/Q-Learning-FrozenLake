"""
q_learning.py
Q-learning agent for Gymnasium FrozenLake-v1.

Requirements:
  pip install gymnasium pygame numpy matplotlib imageio

Usage:
  python q_learning.py
"""


import os
import json
from datetime import datetime

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

# -----------------------
# Config / hyperparams
# -----------------------
ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False 
NUM_EPISODES = 25000
MAX_STEPS_PER_EPISODE = 100
LEARNING_RATE = 0.8          # alpha
DISCOUNT_FACTOR = 0.95       # gamma
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# Environment
# -----------------------
env = gym.make(ENV_ID, desc=None, map_name="4x4", is_slippery=IS_SLIPPERY)  # default render_mode=None
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f"states={n_states}, actions={n_actions}")

# -----------------------
# Q-table init
# -----------------------
q_table = np.zeros((n_states, n_actions))

# -----------------------
# Helpers
# -----------------------
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_table[state]))

# -----------------------
# Training loop
# -----------------------
epsilon = EPSILON_START
rewards_all_episodes = []
reward_moving_avg = []
print_every = max(1, NUM_EPISODES // 10)

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward
        if done:
            break
        
    # decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    rewards_all_episodes.append(total_reward)

    # track moving average for diagnostics
    if episode % 100 == 0:
        avg_recent = np.mean(rewards_all_episodes[-100:])
        reward_moving_avg.append(avg_recent)

    if episode % print_every == 0 or episode == 1:
        print(f"Episode {episode}/{NUM_EPISODES} - eps={epsilon:.4f} avg_last_100={np.mean(rewards_all_episodes[-100:]):.4f}")

# -----------------------
# Evaluation
# -----------------------
def evaluate(q_table, n_episodes=1000):
    total_rewards = 0
    eval_env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY)
    for _ in range(n_episodes):
        state, info = eval_env.reset()
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_rewards += reward
    eval_env.close()
    return total_rewards / n_episodes

eval_score = evaluate(q_table, n_episodes=1000)
print(f"Average reward over 1000 evaluation episodes: {eval_score:.4f}")

# -----------------------
# Save artifacts
# -----------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
np.save(os.path.join(SAVE_DIR, f"q_table_{timestamp}.npy"), q_table)
history = {
    "env": ENV_ID,
    "is_slippery": IS_SLIPPERY,
    "num_episodes": NUM_EPISODES,
    "learning_rate": LEARNING_RATE,
    "discount_factor": DISCOUNT_FACTOR,
    "epsilon_start": EPSILON_START,
    "epsilon_min": EPSILON_MIN,
    "epsilon_decay": EPSILON_DECAY,
    "eval_score": float(eval_score),
    "rewards_all_episodes": rewards_all_episodes,
    "reward_moving_avg": reward_moving_avg,
}
with open(os.path.join(SAVE_DIR, f"training_history_{timestamp}.json"), "w") as f:
    json.dump(history, f)

print(f"Saved Q table and history to {SAVE_DIR}/")

# -----------------------
# Plot learning curve
# -----------------------
plt.figure(figsize=(8, 4))
episodes = np.arange(len(reward_moving_avg)) * 100 + 100
plt.plot(episodes, reward_moving_avg)
plt.xlabel("Episode")
plt.ylabel("Average reward (last 100)")
plt.title("Learning curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"learning_curve_{timestamp}.png"))
print("Saved learning curve plot.")

# -----------------------
# Record demo GIF
# -----------------------
def record_gif(q_table, filename="demo.gif", n_episodes=3):
    demo_env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY, render_mode="rgb_array")
    frames = []
    for episode in range(n_episodes):
        state, info = demo_env.reset()
        done = False
        img = demo_env.render()
        frames.append(img)

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated
            img = demo_env.render()
            frames.append(img)

    demo_env.close()
    path = os.path.join(SAVE_DIR, filename)
    imageio.mimsave(path, frames, fps=5)
    print(f"Saved demo gameplay to {path}")

record_gif(q_table, filename=f"demo_{timestamp}.gif")