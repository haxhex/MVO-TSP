import numpy as np
import random
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(point1, point2):
    # Calculates the Euclidean distance between two points
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Initialize a random TSP problem
def create_tsp_problem(num_cities):
    # Creates a random TSP problem with a given number of cities
    cities = [np.random.randint(10, size=2) for _ in range(num_cities)]
    return cities

# Generate a random tour for the TSP problem
def random_tour(num_cities):
    # Generates a random tour for the TSP problem
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

# Calculate the total distance of a TSP tour
def total_distance(cities, tour):
    # Calculates the total distance of a TSP tour
    return sum(euclidean_distance(cities[tour[i]], cities[tour[i - 1]]) for i in range(len(tour)))

# MVO algorithm function
def mvo_tsp(cities, population_size, num_iterations, w_ep_initial=1):
    # Applies the MVO algorithm to solve the TSP problem
    num_cities = len(cities)
    # Initialize a population of random tours
    universes = [random_tour(num_cities) for _ in range(population_size)]
    # Calculate the fitness of each tour
    fitness = [total_distance(cities, universe) for universe in universes]
    # Initialize the exploration probability
    w_ep = w_ep_initial

    # Run the MVO algorithm for a fixed number of iterations
    for t in range(num_iterations):
        # Find the best and worst fitness values in the population
        best_fitness = min(fitness)
        worst_fitness = max(fitness)
        # Normalize the fitness values
        if worst_fitness == best_fitness:
            norm_fitness = [1.0] * len(fitness)
        else:
            norm_fitness = [(worst_fitness - fit) / (worst_fitness - best_fitness) for fit in fitness]
        # Calculate the probability of each universe to act as a wormhole
        wormhole_probs = [norm_fit / sum(norm_fitness) for norm_fit in norm_fitness]
        # Calculate the cumulative probability of each universe to act as a wormhole
        total_wormhole_probs = [sum(wormhole_probs[0:i + 1]) for i in range(population_size)]

        # Perform the wormhole operation for each universe
        for i in range(population_size):
            # Randomly choose to perform the wormhole operation or not
            rand_prob = random.random()
            if rand_prob < w_ep:
                # Select a universe to perform the wormhole operation
                try:
                    selected_universe_index = next(i for i, prob in enumerate(total_wormhole_probs) if prob >= rand_prob)
                except StopIteration: 
                    selected_universe_index = np.random.randint(0, len(total_wormhole_probs))
            else:
                selected_universe_index = i

            # Perform the wormhole operation to create a new universe
            new_universe = universes[selected_universe_index].copy()
            idx1, idx2 = random.sample(range(num_cities), 2)
            new_universe[idx1], new_universe[idx2] = new_universe[idx2], new_universe[idx1]

            # Calculate the fitness of the new universe
            new_fitness = total_distance(cities, new_universe)

            # Update the current universe if the new universe has better fitness
            if new_fitness < fitness[i]:
                universes[i] = new_universe
                fitness[i] = new_fitness

        # Update the exploration probability
        w_ep = w_ep * (1 - t / num_iterations)

    # Find the best solution in the population
    best_solution_index = fitness.index(min(fitness))
    best_solution = universes[best_solution_index]
    best_solution_fitness = fitness[best_solution_index]

    # Return the best solution and its fitness value
    return best_solution, best_solution_fitness

# Plotting function
def plot_tour(cities, best_tour):
    # Plots a TSP tour using Matplotlib
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red', zorder=1)
    plt.plot([cities[city][0] for city in best_tour + [best_tour[0]]], [cities[city][1] for city in best_tour + [best_tour[0]]], zorder=0)
    for i, city in enumerate(cities):
        # Adds annotations to the plot indicating the index of each city
        plt.annotate("v"+str(i), xy=(city[0], city[1]), xytext=(5, 5), textcoords='offset points')
    plt.show()

# This plot displays the distances between the cities in the TSP problem.
# It can be used to verify that the distance calculations are correct.
# The plot is particularly useful when the number of cities is relatively small, 
# as it allows for a clearer visualization of the distances.
def plot_tour2(cities, best_tour):
    # Plots a TSP tour using Matplotlib
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red', zorder=1)
    for i in range(len(best_tour)):
        # Get the coordinates of the current city and its predecessor
        current_city = cities[best_tour[i]]
        prev_city = cities[best_tour[i-1]]
        # Calculate the distance between the current city and its predecessor
        distance = euclidean_distance(current_city, prev_city)
        # Calculate the midpoint between the current city and its predecessor
        midpoint = ((current_city[0] + prev_city[0]) / 2, (current_city[1] + prev_city[1]) / 2)
        # Add a line between the current city and its predecessor
        plt.plot([current_city[0], prev_city[0]], [current_city[1], prev_city[1]], 'k-', zorder=0)
        # Add an annotation with the distance at the midpoint between the current city and its predecessor
        plt.annotate(str(round(distance, 2)), xy=midpoint, xytext=(5, 5), textcoords='offset points')
        # Add an annotation with the city index at the current city
        plt.annotate("v"+str(best_tour[i]), xy=current_city, xytext=(5, 5), textcoords='offset points')
    plt.show()

# Example usage
num_cities = 10
cities = create_tsp_problem(num_cities)
population_size = 100
num_iterations = 1000
best_tour, best_distance = mvo_tsp(cities, population_size, num_iterations)
print("Best tour:", best_tour)
print("Best distance:", best_distance)
plot_tour(cities, best_tour)
# plot_tour2(cities, best_tour)
plt.show()