"""
Test the performance of the best genome produced by evolve.py.
"""

import os
import pickle

import neat
import gymnasium as gym
import numpy as np

# load the winner
with open("winner", "rb") as f:
    c = pickle.load(f)

print("Loaded genome:")
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(observation[0]))
print("    x_dot = {0:.4f}".format(observation[1]))
print("    theta = {0:.4f}".format(observation[2]))
print("theta_dot = {0:.4f}".format(observation[3]))
print()

done = False
while not done:
    action = np.argmax(net.activate(observation))
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated


print()
print("Final conditions:")
print("        x = {0:.4f}".format(observation[0]))
print("    x_dot = {0:.4f}".format(observation[1]))
print("    theta = {0:.4f}".format(observation[2]))
print("theta_dot = {0:.4f}".format(observation[3]))
print()
