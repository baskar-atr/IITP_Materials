from collections import deque
import random
from datetime import datetime


def print_matrix(state):
    for row in state:
        print(" ".join(map(str, row)))
    print()


def get_blank_cell(currentstate):
    # It is a Generator function, first finds the rows with its index,
    # then finds the columns with index,
    # then filter, if column value matches B
    result = next(
        (rowIndex, colIndex) for rowIndex, row in enumerate(currentstate) for colIndex, col in enumerate(row) if
        col == 'B')
    return result


def total_inversions(state):
    flat_state = [num for row in state for num in row if num != 'B']
    inversions = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1

    return inversions


def is_solvable(state):
    inversions = total_inversions(state)
    # For a 3x3 grid, a state is solvable if the number of inversions is even
    return inversions % 2 == 0


def getAdjacentCells(currentstate):
    neighbours = []
    # Get the empty or B Tile indexes
    zero_row, zero_col = get_blank_cell(currentstate)
    # Calculate the neighbours by adding/subtracting the index value for the movement
    for dr, dc, action in [(-1, 0, 'up'), (0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left')]:
        new_row, new_col = zero_row + dr, zero_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [row[:] for row in currentstate]
            new_state[zero_row][zero_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[
                zero_row][zero_col]
            neighbours.append((new_state, action))
    return neighbours


def get_initial_state():
    numbers = list(range(1, 9))
    random.shuffle(numbers)
    numbers.append('B')
    random_state = [numbers[i:i + 3] for i in range(0, 9, 3)]
    # Check if the state is solvable, regenerate if not
    while not is_solvable(random_state):
        random.shuffle(numbers)
        random_state = [numbers[i:i + 3] for i in range(0, 9, 3)]
    return random_state


def bfs(initial_state, goal_state):
    startTime = datetime.now()
    print('Starting BFS..')
    iteration = 0
    visited = set()
    queue = deque([(initial_state, [])])

    while queue:
        iteration += 1
        state, path = queue.popleft()
        visited.add(tuple(map(tuple, state)))

        if state == goal_state:
            print("BFS Goal Reached!")
            print('BFS ToTal Step Taken is :', len(path))
            print('BFS Total Iterations Taken is :', iteration)
            endTime = datetime.now() - startTime
            print(f'BFS Total Time Taken to attain the goal state in {endTime.total_seconds()} seconds')
            # print("BFS Path to Goal:")
            # print('->'.join(path))
            return path, iteration

        neighbours = getAdjacentCells(state)
        for neighbour, action in neighbours:
            if tuple(map(tuple, neighbour)) not in visited:
                visited.add(tuple(map(tuple, neighbour)))
                queue.append((neighbour, path + [action]))
    print("BFS Goal not reachable. So retrying with new random sample")
    return None, iteration


def dfs(initial_state, goal_state):
    startTime = datetime.now()
    print('Starting DFS..')
    stack = [(initial_state, [])]
    visited = set()
    visited.add(tuple(map(tuple, initial_state)))
    iteration = 0

    while stack:
        iteration += 1
        current_state, path = stack.pop()  # FIFO
        if len(path) < 20000:
            if current_state == goal_state:
                print("DFS Goal Reached!")
                print('DFS ToTal Step Taken is :', len(path))
                print('DFS Total Iterations Taken is :', iteration)
                endTime = datetime.now() - startTime
                print(f'DFS Total Time Taken to attain the goal state in {endTime.total_seconds()} seconds')
                return path, iteration
            neighbours = getAdjacentCells(current_state)
            for neighbour, action in neighbours:
                if tuple(map(tuple, neighbour)) not in visited:
                    visited.add(tuple(map(tuple, neighbour)))
                    stack.append((neighbour, path + [action]))
        else:
            print("DFS reached 20000 steps,So retrying with new random sample")
            return None, iteration
    return path, iteration


def search(goal_state, attempt=0):
    attempt += 1
    # Generate Random Initial State
    initial_state = get_initial_state()
    print(f'Attempt {attempt} : Random Initial State:')
    print_matrix(initial_state)

    bfs_path, iteration = bfs(initial_state, goal_state)
    if not bfs_path:
        search(goal_state, attempt)
    dfs_path, iteration = dfs(initial_state, goal_state)

    if not dfs_path:
        search(goal_state, attempt)
    return None


# Goal State
goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 'B']
]
print("Goal State:")
print_matrix(goal_state)

search(goal_state)
