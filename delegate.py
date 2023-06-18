"""
as the duty of implementor:
    1) prepare your code in initialize method.
    2) cleanup after your code in terminate method.
    3) replace entry point of your algorithm in native_process method
    4) you must return a 1D array which is your proposed sequence of nodes for traveling through and an integer as
       total path cost of proposed path (permutation).

    exp:
        input :
            np.array([
                [0, 5, 4, 10],
                [5, 0, 8, 5],
                [4, 8, 0, 3],
                [10, 5, 3, 0]
            ])

        output:
            ([0, 1, 3, 2] , 17)

    *** input of your code will be a 2D numpy array where each index represents edge connecting nodes i and j.
    *** if two nodes do not have a edge connecting them directly, corresponding index will be set to -1.
"""
#from .exact.dynamic_programming import solve_tsp_dynamic_programming
from mvo_tsp import euclidean_distance, mvo_tsp, create_tsp_problem
import numpy as np

def native_id():
    # return your project name, team name or some unique id
    return "MVO-TSP"


def initialize():
    # setup initialization code in here
    # this method is called only once at the beginning benchmark
    return


def terminate():
    # setup cleanup code in here
    # this method is called only once at the end of benchmark
    return


def convert_input(raw_input):
    # convert our input here
    v = len(raw_input)
    converted_input = np.zeros((v, v))
    for i in range(v):
        for j in range(v):
            converted_input[i][j] = euclidean_distance(raw_input[i], raw_input[j])
    return converted_input

def convert_output(native_output):
    # convert your output here
    return native_output

# what is data?
def native_process(data):
    # delete line below, replace it with your own code
    # your return type must be a tuple consisting of permutation and total distance
    num_cities = 30
    population_size = 50
    num_iterations = 10
    w_ep = 1

    # Create a random TSP problem
    cities = create_tsp_problem(num_cities)

    # Apply MVO to the TSP problem
    permutation, total_distance = mvo_tsp(cities, population_size, num_iterations, w_ep)
    return permutation, total_distance

# test converting input
# cities = [np.random.rand(2) for _ in range(5)]
# print("raw input be like :\n", cities)
# print("converted input be like :\n", convert_input(cities))

# what is data?
data = 0
print(native_process(data))