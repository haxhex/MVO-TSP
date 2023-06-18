import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Initialize a random TSP problem
def create_tsp_problem(num_cities):
    # cities = [np.random.rand(2) for _ in range(num_cities)]
    # integer(less than 10) input
    cities = [np.random.randint(10, size=2) for _ in range(num_cities)]
    return cities

# Calculate the total distance of a TSP tour
def total_distance(cities, tour):
    return sum(euclidean_distance(cities[tour[i]], cities[tour[i - 1]]) for i in range(len(tour)))

# Generate a random tour for the TSP problem
def random_tour(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

# MVO algorithm function
def mvo_tsp(cities, population_size, num_iterations, w_ep):
    num_cities = len(cities)

    # Initialize the universes (solutions)
    universes = [random_tour(num_cities) for _ in range(population_size)]

    # Calculate the fitness of the universes
    fitness = [total_distance(cities, universe) for universe in universes]

    # Main loop of the MVO algorithm
    for t in range(num_iterations):
        # Calculate the best and worst fitness values
        best_fitness = min(fitness)
        worst_fitness = max(fitness)

        # Calculate the normalized fitness values
        norm_fitness = [(worst_fitness - fit) / (worst_fitness - best_fitness) for fit in fitness]

        # Calculate the updated wormhole probabilities
        wormhole_probs = [norm_fit / sum(norm_fitness) for norm_fit in norm_fitness]

        # Calculate the total wormhole probabilities
        total_wormhole_probs = [sum(wormhole_probs[0:i + 1]) for i in range(population_size)]

        # Update the universes (solutions)
        for i in range(population_size):
            # Calculate the random probability
            rand_prob = random.random()

            # Find the selected universe
            selected_universe_index = next(i for i, prob in enumerate(total_wormhole_probs) if prob >= rand_prob)

            # Create a new universe by swapping two cities in the selected universe
            new_universe = universes[selected_universe_index].copy()
            idx1, idx2 = random.sample(range(num_cities), 2)
            new_universe[idx1], new_universe[idx2] = new_universe[idx2], new_universe[idx1]

            # Calculate the fitness of the new universe
            new_fitness = total_distance(cities, new_universe)

            # Update the universe and its fitness if the new universe is better
            if new_fitness < fitness[i]:
                universes[i] = new_universe
                fitness[i] = new_fitness

            # Update the wormhole existence probability
            w_ep = w_ep * (1 - t / num_iterations)

    # Find the best solution
    best_solution_index = fitness.index(min(fitness))
    best_solution = universes[best_solution_index]
    best_solution_fitness = fitness[best_solution_index]

    return best_solution, best_solution_fitness

def plot_tour(cities, best_tour):
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red', zorder=1)
    plt.plot([cities[city][0] for city in best_tour + [best_tour[0]]], [cities[city][1] for city in best_tour + [best_tour[0]]], zorder=0)
    for i, city in enumerate(cities):
        plt.annotate("v"+str(i), xy=(city[0], city[1]), xytext=(5, 5), textcoords='offset points')
    plt.show()