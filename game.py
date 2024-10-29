import streamlit as st
import pygame
import numpy as np
import random
import time
from pygame.surfarray import array3d

# Streamlit setup
st.title("Reinforcement Learning Agent Visualization")

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
ROWS = HEIGHT // GRID_SIZE
COLS = WIDTH // GRID_SIZE

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Agent and environment setup
agent_pos = [0, 0]
target_pos = [ROWS - 1, COLS - 1]
obstacles = [(5, 5), (6, 5), (5, 6), (10, 10), (11, 10), (10, 11)]

# Initialize Q-table
q_table = np.zeros((ROWS, COLS, 4))  # 4 actions: up, down, left, right

# Define actions
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Helper functions
def draw_grid(win):
    win.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(win, BLACK, rect, 1)
    pygame.draw.rect(win, RED, (target_pos[1] * GRID_SIZE, target_pos[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    for obstacle in obstacles:
        pygame.draw.rect(win, BLACK, (obstacle[1] * GRID_SIZE, obstacle[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def draw_agent(win):
    pygame.draw.rect(win, GREEN, (agent_pos[1] * GRID_SIZE, agent_pos[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def get_reward(pos):
    if pos == target_pos:
        return 100  # Reward for reaching the target
    elif pos in obstacles:
        return -100  # Penalty for hitting an obstacle
    else:
        return -1  # Small penalty for each step taken

def move_agent(action):
    new_pos = [agent_pos[0] + actions[action][0], agent_pos[1] + actions[action][1]]
    if 0 <= new_pos[0] < ROWS and 0 <= new_pos[1] < COLS:  # Check if within grid bounds
        return new_pos
    return agent_pos

# Main loop for training
def train_agent(episodes=100):
    global agent_pos, epsilon
    image_placeholder = st.empty()  # Placeholder for updating the image
    for episode in range(episodes):
        agent_pos = [0, 0]
        done = False
        while not done:
            # Choose action
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(actions.keys()))  # Explore
            else:
                action = np.argmax(q_table[agent_pos[0], agent_pos[1]])  # Exploit

            # Take action
            new_pos = move_agent(action)
            reward = get_reward(new_pos)
            old_value = q_table[agent_pos[0], agent_pos[1], action]
            next_max = np.max(q_table[new_pos[0], new_pos[1]])

            # Q-learning formula
            q_table[agent_pos[0], agent_pos[1], action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            agent_pos = new_pos

            # Check if episode is done
            if agent_pos == target_pos or agent_pos in obstacles:
                done = True

            # Draw environment
            WIN = pygame.Surface((WIDTH, HEIGHT))
            draw_grid(WIN)
            draw_agent(WIN)

            # Convert Pygame surface to image for Streamlit
            img_array = array3d(WIN)
            image_placeholder.image(img_array, caption="Agent Training", width=400)

            # Decrease exploration rate over time
            epsilon = max(0.01, epsilon * 0.995)

            # Add delay for live visualization effect
            time.sleep(0.05)

# Start training
if st.button("Start Training"):
    train_agent(episodes=1000)

st.write("Click 'Start Training' to train the agent and visualize the process.")