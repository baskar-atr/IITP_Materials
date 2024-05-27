import random
import copy


class Hill_Search:
    def __init__(self, state=None):
        self.state = state if state is not None else [['T1', 'T2', 'T3'], ['T4', 'T5', 'T6'], ['T7', 'T8', 'B']]

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.state])

    def goal_test(self):
        return self.state == [['T1', 'T2', 'T3'], ['T4', 'T5', 'T6'], ['T7', 'T8', 'B']]

    def get_blank_position(self):
        # It is a Generator function, first finds the rows with its index,
        # then finds the columns with index,
        # then filter, if column value matches B
        blank_pos = next(
            (rowIndex, colIndex) for rowIndex, row in enumerate(self.state) for colIndex, col in enumerate(row) if
            col == 'B')
        return blank_pos

    def get_neighbors(self):
        neighbors = []
        # find the Blank tiles indices
        i, j = self.get_blank_position()
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for move in moves:
            ni, nj = i + move[0], j + move[1]
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = copy.deepcopy(self.state)
                new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                neighbors.append(Hill_Search(new_state))
        return neighbors


def displaced_heuristics(state):
    displaced_tiles = sum(state[i][j] != goal_state[i][j] for i in range(3) for j in range(3))
    print("Number of displaced tiles : ", displaced_tiles)
    return displaced_tiles


def manhattan_heuristics(state):
    manhattan_distance = sum(abs(i - (int(state[i][j][1]) - 1) // 3) + abs(j - (int(state[i][j][1]) - 1) % 3)
                             for i in range(3) for j in range(3) if state[i][j] != 'B')
    print('Manhattan distance is:', manhattan_distance)
    return manhattan_distance


def is_solvable(state):
    inversion_count = sum(1 for i in range(9) for j in range(i + 1, 9)
                          if state[i] != 'B' and state[j] != 'B' and state[i] > state[j])
    return inversion_count % 2 == 0


def create_random_state():
    tiles = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'B']
    random.shuffle(tiles)
    return [tiles[i:i + 3] for i in range(0, 9, 3)]


def hill_climbing_search(initial_state, heuristic):
    total_states_explored = 0
    path = [initial_state]  # Initialize path with initial state
    current = Hill_Search(initial_state)
    # Current State is not goal state and total states explores is less than 100 times
    while not current.goal_test() and total_states_explored < 100:
        total_states_explored += 1
        # Find the neighbours by swapping the tiles up,down,left,right
        neighbors = current.get_neighbors()
        print('State Number:', total_states_explored)
        # Dictionary has Neighbour/successor states with its heuristics
        neighbor_heuristics = [(neighbor, heuristic(neighbor.state)) for neighbor in neighbors]
        # Check if current state's heuristic value is greater than are equal to all neighbours heuristic values
        if all(neighbor_heuristic[1] >= heuristic(current.state) for neighbor_heuristic in neighbor_heuristics):
            print(f'Local Minima reached with heuristic Value {heuristic(current.state)} at state:')
            print(current)
            return "Failure", initial_state, total_states_explored, "Local Minima Reached",path
        # Find the Minimum heuristic value among the neighbours/successors
        best_neighbor, best_score = min(neighbor_heuristics, key=lambda x: x[1])
        current = best_neighbor
        path.append(current.state)
    if current.goal_test():
        return "Success", initial_state, total_states_explored, "Goal Reached",path
    else:
        return "Failure", initial_state, total_states_explored - 1, "Maximum States Explored", path[:-1]


if __name__ == "__main__":
    # Given Goal State
    goal_state = [['T1', 'T2', 'T3'], ['T4', 'T5', 'T6'], ['T7', 'T8', 'B']]
    # Sample successful states to verify
    sample_input_state1 = [['T1', 'T2', 'T3'], ['T4', 'T5', 'T6'], ['T7', 'B', 'T8']]
    sample_input_state2 = [['T1', 'T2', 'T3'], ['T4', 'B', 'T6'], ['T7', 'T5', 'T8']]
    success_states = [sample_input_state1, sample_input_state2]

    solvable = False
    attempts = 0
    while not solvable and attempts < 100:
        # Create Random initial State
        initial_state = create_random_state()
        # Check solvable or not using * Puzzle theory of Inversion
        solvable = is_solvable([num for row in initial_state for num in row])
        attempts += 1

    if solvable:
        success_states.append(initial_state)
        for initial_state in success_states:
            print("**********************************************************************************************")
            print("Hill Climbing Search started, Initial State is,")
            print(initial_state)
            # Solving using h1
            print("Solving using h1 (number of tiles displaced from their destined position)")
            result, initial_state, total_states_explored, termination_reason ,path= hill_climbing_search(initial_state, displaced_heuristics)
            print('Optimal path using h1:',path)
            if result == "Success":
                print("Start State:")
                print(Hill_Search(initial_state))
                print("Goal State:")
                print(Hill_Search(goal_state))
                print("Status is Success,Total number of states explored:", total_states_explored)
            else:
                print("Start State:")
                print(Hill_Search(initial_state))
                print("Goal State:")
                print(Hill_Search(goal_state))
                print("Total number of states explored before termination:", total_states_explored)
                print("Status is Failed, Termination Reason:", termination_reason)
            print('--------------------------------------------------------------------------------------------')
            # Solving using h2
            print("Solving using h2 (sum of the Manhattan distance of each tile from the goal position)")
            result, initial_state, total_states_explored, termination_reason, path = hill_climbing_search(initial_state, manhattan_heuristics)
            print('Optimal path using h2:', path)
            if result == "Success":
                print("Start State:")
                print(Hill_Search(initial_state))
                print("Goal State:")
                print(Hill_Search(goal_state))
                print(f'Status is Success, Total number of states explored: {total_states_explored}')
            else:
                print("Start State:")
                print(Hill_Search(initial_state))
                print("Goal State:")
                print(Hill_Search(goal_state))
                print("Total number of states explored before termination:", total_states_explored)
                print(f'Status is Failed,Termination Reason: {termination_reason}')
    else:
        print("Failed after 100 attempts also.")
