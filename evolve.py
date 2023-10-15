"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle
import numpy as np
import neat
import gymnasium as gym
import visualize

runs_per_net = 5


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    env = gym.make("CartPole-v1")
    for runs in range(runs_per_net):
        observation, info = env.reset()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        while not done:
            action = np.argmax(net.activate(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            fitness += reward
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
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

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open("winner", "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="fitness.svg")
    visualize.plot_species(stats, view=True, filename="speciation.svg")

    node_names = {-1: "x", -2: "dx", -3: "theta", -4: "dtheta", 0: "control"}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner-network.gv",
    )


if __name__ == "__main__":
    run()
