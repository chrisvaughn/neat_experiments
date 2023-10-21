"""
Test the performance of the best genome produced by evolve.py.
"""

import os
import pickle
import sys

import gymnasium as gym
import neat

from evolve import eval_genomes

if sys.argv[1] == "best":
    print("Using best instead of winner.")
    with open("best", "rb") as f:
        winner = pickle.load(f)
else:
    # load the winner
    with open("winner", "rb") as f:
        winner = pickle.load(f)

print("Loaded genome:")
print(winner)

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

net = neat.nn.FeedForwardNetwork.create(winner, config)
env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset()

done = False
while not done:
    action = net.activate(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
