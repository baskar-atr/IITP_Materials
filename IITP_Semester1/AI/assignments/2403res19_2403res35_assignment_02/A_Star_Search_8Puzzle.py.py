import numpy as np
import heapq
import random
import time


# Heuristic functions
def h1(state):
    # Heuristic h1: Zero (always returns 0)
    return 0


def h2(state):
    # Heuristic h2: Number of misplaced tiles (excluding the empty space)
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    return np.sum(state != goal_state) - 1


def h3(state):
    # Heuristic h3: Manhattan distance
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:
                goal_position = np.where(goal_state == state[i, j])
                distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance


def h4(state):
    # Heuristic h4: Custom heuristic (h(n) > h*(n))
    # In this case, just add a constant value to h3
    return h3(state) + 10


def astar_search(start_state, heuristic):
    # this function gets the function as input param : heuristic
    # Based on the value we pass h1/h2/h3/h4 it's going to call that function
    def calculate_cost(state, steps):
        # Calculate total cost (f(n)) for A* search
        return heuristic(state) + steps

    def getNeighbours(state):
        # Generate successor states
        neighbours = []
        # find the Grids indices for the empty/zero value
        # Here we consider empty as 0
        zero_position = np.where(state == 0)
        zero_x, zero_y = zero_position[0][0], zero_position[1][0]
        # zero_x, zero_y are the indices of the blank cell
        # Define possible moves (up, down, left, right)
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for move in moves:
            new_x, new_y = zero_x + move[0], zero_y + move[1]
            # boundary conditions, so that the shifting will be with in 3x3 matrix
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                # copy the state
                new_state = state.copy()
                # swap the empty cell with neighbour
                new_state[zero_x, zero_y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[zero_x, zero_y]
                neighbours.append(new_state)
        return neighbours

    # Initialize cost tracker and priority queue
    visited = set()  # To track visited states
    open_list = []  # Priority queue for the open list
    iterations = 0  # Count iterations

    # Flatten the start_state for set operations so that its hashable.
    start_state_flat = tuple(start_state.flatten())
    visited.add(start_state_flat)
    # we use heapq as priority Queue, Efficient implementation of priority queue using a binary heap
    # store the
    heapq.heappush(open_list, (calculate_cost(start_state, 0), start_state_flat, start_state, 0))

    # A* search
    while open_list:
        # heapq.heappop is poping the element from open_lsit (priority Queue)
        # Since newcost not required we keep teh variable as _
        _, current_state_flat, current_state, steps = heapq.heappop(open_list)
        iterations += 1

        # Check if current state is goal state
        if np.array_equal(current_state, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])):
            return current_state, steps, iterations  # Return goal state, steps, and iterations

        # Get successors and calculate costs
        neighbours = getNeighbours(current_state)
        for neighbour in neighbours:
            neighbour_flat = tuple(neighbour.flatten())
            if neighbour_flat in visited:
                continue  # Skip already visited states
            new_steps = steps + 1
            new_cost = calculate_cost(neighbour, new_steps)

            # Add to visited set and open list
            visited.add(neighbour_flat)
            heapq.heappush(open_list, (new_cost, neighbour_flat, neighbour, new_steps))

    # If goal is not reached, return None
    return None, None, iterations


# Function to generate a random valid state
def generate_random_state():
    state = list(range(9))
    random.shuffle(state)
    return np.array(state).reshape(3, 3)


# Function to retry until goal state is reached
def astar_search_with_retry(heuristic):
    # continue until it reaches the goal state
    while True:
        initial_state = generate_random_state()
        goal_state, steps, iterations = astar_search(initial_state, heuristic)
        # Return if goal state is reached
        if goal_state is not None:
            return initial_state, goal_state, steps, iterations


# Main function to evaluate all heuristics
def evaluate_heuristics():
    # Four Heuristic functions defined according to the assignment
    heuristics = [h1, h2, h3, h4]

    # Initialize the variables which will store the minimum cost,iterations,time and its names
    # Minimum Cost defined as Infinitive
    min_cost = float('inf')
    min_heuristic_cost_fn = None
    min_time = float('inf')
    min_heuristic_time_fn = None
    min_iterations = float('inf')
    min_heuristic_iterations_fn = None

    for heuristic in heuristics:
        print(f'Starting Heuristic:{heuristic.__name__}')
        start_time = time.time()
        start_state, goal_state, steps, iterations = astar_search_with_retry(heuristic)
        end_time = time.time()
        print("Initial State:")
        print(start_state)
        print("Goal State:")
        print(goal_state)
        print(f'Heuristic:{heuristic.__name__} took {steps} steps, {iterations} Iterations and took {round(end_time - start_time, 4)} seconds to reach the goal')
        print("Total Cost (Steps + Heuristic Value):", steps + heuristic(start_state))
        print(f'{heuristic.__name__} completed ************************************************')
        if steps + heuristic(start_state) < min_cost:
            min_cost = steps + heuristic(start_state)
            min_heuristic_cost_fn = heuristic.__name__
        if end_time - start_time < min_time:
            min_time = end_time - start_time
            min_heuristic_time_fn = heuristic.__name__
        if iterations < min_iterations:
            min_iterations = iterations
            min_heuristic_iterations_fn = heuristic.__name__

    print(f'Minimum Total Cost among Heuristics:{min_cost} by Heuristic Fn :{min_heuristic_cost_fn}')
    print("Minimum Time Taken:", round(min_time, 4), "seconds (Heuristic:", min_heuristic_time_fn, ")")
    print("Minimum Iterations:", min_iterations, "(Heuristic:", min_heuristic_iterations_fn, ")")


# Evaluate all heuristics functions
evaluate_heuristics()
