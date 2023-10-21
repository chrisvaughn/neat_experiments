"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle
import sys

import gymnasium as gym
import neat
import numpy as np

import visualize

runs_per_net = 20
generations = 500

restore = False
checkpoint_filename = "neat-checkpoint-179"
if len(sys.argv) > 1 and sys.argv[1] == "restore":
    restore = True


class SaveBestGenome(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        with open("best", "wb") as f:
            pickle.dump(best_genome, f)


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    env = gym.make("BipedalWalker-v3")
    for runs in range(runs_per_net):
        observation, info = env.reset()
        fitness = 0.0
        done = False
        while not done:
            action = net.activate(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            fitness += reward
        fitnesses.append(fitness)

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

    if restore:
        print(f"Restoring from {checkpoint_filename}")
        pop = neat.checkpoint.Checkpointer.restore_checkpoint(checkpoint_filename)
    else:
        pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    save_best_genome = SaveBestGenome()
    pop.add_reporter(save_best_genome)
    pop.add_reporter(neat.StdOutReporter(True))
    checkpointer = neat.Checkpointer(10, 900)
    pop.add_reporter(checkpointer)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, generations)

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
