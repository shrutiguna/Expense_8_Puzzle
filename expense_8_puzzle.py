#Shruti Gunasekaran -1002162170 
#AI assignment 1
#Coded language- python 3.10.11

import sys
import datetime
from collections import deque
from queue import Queue, PriorityQueue
import hashlib


#Breadth first search


# The StateNode class represents a state in the search tree.
class StateNode:
    def __init__(self, state, parent, action, cost, depth):
        self.state = state  # The current state of the puzzle
        self.parent = parent  # The parent node in the search tree
        self.action = action  # The action taken to reach this state
        self.cost = cost  # The cost to reach this state
        self.depth = depth  # The depth of this node in the search tree

    def __eq__(self, other):
        return self.state == other.state  # Check if two StateNodes are equal by comparing their states.

    def __hash__(self):
        state_bytes = str(self.state).encode('utf-8')  # Generate a hash value for the StateNode based on its state.
        return int(hashlib.sha256(state_bytes).hexdigest(), 16)

# Function to expand a node by generating its successors.
def expand_node(state_node):
    offspring = []
    actions = ["Up", "Down", "Left", "Right"]
    initial_index = state_node.state.index(0)
    row, col = divmod(initial_index, 3)

    for action in actions:
        if action == "Up":
            move_row, move_col = row - 1, col
        elif action == "Down":
            move_row, move_col = row + 1, col
        elif action == "Right":
            move_row, move_col = row, col + 1
        else:
            move_row, move_col = row, col - 1

        if not (0 <= move_row <= 2 and 0 <= move_col <= 2):
            continue

        new_position = move_row * 3 + move_col
        updated_state = list(state_node.state)
        updated_state[initial_index], updated_state[new_position] = updated_state[new_position], updated_state[initial_index]
        cost = state_node.state[new_position]
        new_node = StateNode(tuple(updated_state), state_node, action, cost, state_node.depth + 1)
        offspring.append(new_node)

    return offspring

# Function to fetch successors of a given state node.
def fetch_successors(state_node):
    successors = []
    i, j = get_blank_position(state_node.state)
    if i > 0:
        state = swap(state_node.state, i, j, i-1, j)
        successors.append(StateNode(state, "Move {} Down".format(state[i][j]), state_node.cost+1, state_node.depth+1, state_node))
    if i < 2:
        state = swap(state_node.state, i, j, i+1, j)
        successors.append(StateNode(state, "Move {} Up".format(state[i][j]), state_node.cost+1, state_node.depth+1, state_node))
    if j > 0:
        state = swap(state_node.state, i, j, i, j-1)
        successors.append(StateNode(state, "Move {} Right".format(state[i][j]), state_node.cost+1, state_node.depth+1, state_node))
    if j < 2:
        state = swap(state_node.state, i, j, i, j+1)
        successors.append(StateNode(state, "Move {} Left".format(state[i][j]), state_node.cost+1, state_node.depth+1, state_node))
    return successors

# Function to get the position of the blank tile (0) in the puzzle.
def get_blank_position(state):
    for r, row in enumerate(state):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    raise ValueError("The node is not valid because there is no blank tile (0) found")

# Function to swap the blank tile with an adjacent tile.
def swap(state, r1, c1, r2, c2):
    state = [list(row) for row in state]
    state[r1][c1], state[r2][c2] = state[r2][c2], state[r1][c1]
    return tuple(map(tuple, state))

# Function to perform Breadth-First Search (BFS) to find the goal state.
def bfs_search(start, goal):
    visited = set()  # Set to keep track of visited nodes
    q = Queue()  # Queue to hold nodes to be explored
    root = StateNode(start, None, None, 0, 0)  # Create the root node
    q.put(root)  # Add the root node to the queue
    visited.add(root)  # Mark the root node as visited
    popped_count = 0  # Counter for the number of nodes popped from the queue
    max_queue_size = 0  # Track the maximum size of the queue
    solution_found = False  # Flag to indicate if the solution is found
    expanded_count = 0  # Counter for the number of nodes expanded

    generated_count = 0  # Counter for the number of nodes generated
    visited_states = set()  # Set to keep track of visited states

    if dump == 'true':  # Open a file to dump search details if required
        file = open(timestamp, 'a')

    while not q.empty():
        max_queue_size = max(max_queue_size, q.qsize())  # Update the maximum queue size
        state_node = q.get()
        popped_count += 1

        if state_node.state == goal:
            solution_found = True  # Here is where BFS finds a solution
            search_depth = 0
            path = []
            total_cost = 0

            while state_node.parent is not None:
                path.append((state_node.action, state_node.cost))
                total_cost += state_node.cost
                search_depth += state_node.depth
                state_node = state_node.parent

            path.reverse()

            return popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost, path

        if dump == 'true':
            file.write(f"Generating successors to < state = {state_node.state}, action = {state_node.action} g(n) = {state_node.cost}, d = {state_node.depth}, f(n) = {state_node.cost + state_node.depth} >:")

        offspring = expand_node(state_node)  # Generate successors of the current node
        expanded_count += 1
        visited_states.add(state_node.state)  # Mark the current state as visited

        queue_state = "[\n"
        for child in offspring:
            generated_count += 1

            if child not in visited_states:
                q.put(child)  # Add the child node to the queue
                visited.add(child)  # Mark the child node as visited
                if dump == 'true':
                    queue_state += f"< state = {child.state}, action = {child.action}, g(n) = {child.cost}, d = {child.depth}, f(n) = {child.cost + child.depth}, Parent = Pointer to {child.parent.state} >\n"

        if dump == 'true':
            queue_state += "]"
            file.write(f"{len(offspring)} successors generated\nClosed: {list(visited_states)}\tFringe: {queue_state}\n")

    print("No solution found.")
    return None
# Depth First Search

class StateNodeD:
    def __init__(self, state, parent=None, cost=0, action=None, depth=0):
        self.state = state   # Similar definations as done with BFS
        self.parent = parent
        self.cost = cost
        self.action = action
        self.depth = depth
        self.offspring = []

    def add_child(self, child_node):
        self.offspring.append(child_node) # Add a child node to the list of offspring

    def is_leaf(self):
        return len(self.offspring) == 0    # Check if the node is a leaf node (has no children)


    def __eq__(self, other):
        return self.state == other.state # Check if two StateNodes are equal by comparing their states

    def __hash__(self):
        return hash(str(self.state))   # Generate a hash value for the StateNode based on its state


# Function to perform Depth-First Search (DFS) to find the goal state
def dfs_search(start, goal):
    initial_vertex = StateNodeD(start, None, 0, None)  # create root vertex
    max_stack_size = 0 # To track max size of the stack
    expanded_count = 0
    generated_count = 0
    popped_count = 0
    stack = [initial_vertex]  # initialize stack with root vertex
    visited_states = set()  # Set to keep track of visited states

    # Open file for writing if dump mode is on
    if dump == 'true':
        file = open(timestamp, 'a')

    # To search until stack is empty
    while stack:
        max_stack_size = max(max_stack_size, len(stack))  # update max stack size
        state_node = stack.pop()  # get vertex from stack
        popped_count += 1  # update nodes popped count
        path = []  # initialize path list

        # write to file if dump mode is on
        if dump == 'true':
            file.write(f"Expanding: {state_node.state}")

        # check if vertex is goal vertex, here vertex is nothing but the node
        if state_node.state == goal:
            path = []
            while state_node.parent is not None:
                path.append((state_node.action, state_node.cost))
                state_node = state_node.parent
            path.reverse()
            return popped_count, expanded_count, generated_count, max_stack_size, len(path), sum(cost for _, cost in path), path

        # add vertex to visited set and update nodes expanded count
        visited_states.add(tuple(state_node.state))
        expanded_count += 1

        # generate successors for current vertex
        offspring = fetch_successors_dfs(state_node.state)

        # write to file if dump mode is on

        if dump == 'true':
            file.write(f"\nGenerating successors to < state = {state_node.state}, action = {state_node.action} g(n) = {state_node.cost}, d = {len(path)}, f(n) = {state_node.cost + len(path)}, Parent = Pointer to {{{state_node.parent}}}>:")
            file.write(f"\n{len(offspring)} successors generated")
            file.write(f"\nClosed: {list(visited_states)}")
            file.write(f"\nFringe: [")

        # add unexplored successors to stack and update nodes generated count
        for action, cost, child_state in offspring:
            if tuple(child_state) not in visited_states:
                generated_count += 1
                child_node = StateNodeD(child_state, state_node, cost, action)
                stack.append(child_node)
                if dump == 'true':
                    file.write(f"\n\\t< state = {child_state}, action = {action} g(n) = {state_node.cost + cost}, d = {len(path) + 1}, f(n) = {state_node.cost + cost + len(path) + 1}, Parent = Pointer to {{{state_node.state}}}>")
        if dump == 'true':
            file.write("\n]")

    return None

# Function to fetch successors of a given state node for DFS
def fetch_successors_dfs(state):
    offspring = []
    initial_index = state.index(0)
    moves = [
        ('Down', initial_index - 3),
        ('Up', initial_index + 3),
        ('Right', initial_index - 1),
        ('Left', initial_index + 1),
    ]
    for move, new_position in moves:
        if new_position < 0 or new_position >= len(state):
            continue
        child_state = state[:]
        child_state[initial_index], child_state[new_position] = child_state[new_position], child_state[initial_index]
        offspring.append((move, child_state[initial_index], child_state))
    return offspring

# UCS

# Function to locate the blank tile (0) in the puzzle
def locate_blank(state):
    blank_positions = [(i, j) for i, row in enumerate(state) for j, val in enumerate(row) if val == 0]
    if len(blank_positions) != 1:
        raise ValueError(f"Expected 1 blank position, found {len(blank_positions)}")
    return blank_positions[0]

# Function to fetch possible moves based on the position of the blank tile
def fetch_possible_moves(state):
    ROWS, COLS = 3, 3
    actions = []
    # Find the blank tile location
    blank_row, blank_col = next((r, c) for r in range(ROWS) for c in range(COLS) if state[r][c] == 0)

    # Determine the valid actions based on the blank tile location
    if blank_row > 0:
        actions.append('Up')
    if blank_row < ROWS - 1:
        actions.append('Down')
    if blank_col > 0:
        actions.append('Left')
    if blank_col < COLS - 1:
        actions.append('Right')

    return actions

# Function to generate a new state based on the given action
def generate_new_state(state, action):
    row, col = locate_blank(state)
    updated_state = [row[:] for row in state]
    move_weight = 0
    blank_row, blank_col = locate_blank(state)
    ROWS, COLS = 3, 3
    return_action = ''
    if action == 'Up' and blank_row > 0:
        return_action = "Down"
        updated_state[blank_row][blank_col], updated_state[blank_row-1][blank_col] = updated_state[blank_row-1][blank_col], updated_state[blank_row][blank_col]

    elif action == 'Down' and blank_row < ROWS - 1:
        return_action = "Up"
        updated_state[blank_row][blank_col], updated_state[blank_row+1][blank_col] = updated_state[blank_row+1][blank_col], updated_state[blank_row][blank_col]

    elif action == 'Left' and blank_col > 0:
        return_action = "Right"
        updated_state[blank_row][blank_col], updated_state[blank_row][blank_col-1] = updated_state[blank_row][blank_col-1], updated_state[blank_row][blank_col]

    elif action == 'Right' and blank_col < COLS - 1:
        return_action = "Left"
        updated_state[blank_row][blank_col], updated_state[blank_row][blank_col+1] = updated_state[blank_row][blank_col+1], updated_state[blank_row][blank_col]

    return updated_state, return_action

# Function to perform Uniform Cost Search (UCS) to find the goal state
def ucs_search(initial_state, goal_state):

    expanded_count = 0 # same as previous ones, self-explanatory 
    generated_count = 0
    max_queue_size = 0
    popped_count = 0 # Counter for the number of nodes popped from the priority queue
    search_depth = 0
    total_cost = 0 # Total cost of the path
    path = [] # List to store the path to the goal
    move_weight = [] # List to store the weights of the moves

    priority_queue = PriorityQueue()
    priority_queue.put((0, None, (initial_state, [], [])))  # Initialize the priority queue with the initial state

    visited_states = set()

    # Open a file to dump search details if required
    if dump == 'true':
        file = open(timestamp, 'a')
        file.write(f"Generating successors to < state = {initial_state}, action = {{Start}} g(n) = 0, d = 0, f(n) = 0, Parent = Pointer to {{None}} >:\n")
        file.write(f"{len(fetch_possible_moves(initial_state))} successors generated\n")
        file.write(f"Closed: []\n")

  # Search until the priority queue is empty
    while not priority_queue.empty():
        max_queue_size = max(max_queue_size, priority_queue.qsize())   # Update the maximum queue size

        current_cost, action_state, (current_state, current_path, current_weight) = priority_queue.get()
        popped_count += 1 # Increment the popped count

 # Check if the current state is the goal state
        if goal_state == current_state:
            expanded_count = len(visited_states)
            search_depth = len(current_path)
            move_weight = current_weight
            total_cost = current_cost
            path = current_path

            break

        visited_states.add(tuple(map(tuple, current_state)))
 # Generate possible moves for the current state
        actions = fetch_possible_moves(current_state)

        for action in actions:
            updated_state, action_state = generate_new_state(current_state, action)
            updated_cost = current_cost + updated_state[locate_blank(current_state)[0]][locate_blank(current_state)[1]]
            updated_weight = updated_state[locate_blank(current_state)[0]][locate_blank(current_state)[1]]
            if tuple(map(tuple, updated_state)) not in visited_states:
                priority_queue.put((updated_cost, action_state, (updated_state, current_path + [action_state], current_weight + [updated_weight])))
                generated_count += 1

     # Write to the file if dump mode is on            
            if dump == 'true':
                file.write(f"\nGenerating successors to < state = {current_state}, action = {action_state} g(n) = {current_cost}, d = {len(current_path)}, f(n) = {current_cost}, Parent = Pointer to {{{'None' if len(current_path) == 0 else current_path[-1]}}}>:")
                file.write(f"\n{len(actions)} successors generated")
                file.write(f"\nClosed: {visited_states}")
                file.write("\nFringe: [")
                for item in priority_queue.queue:
                    file.write(f"\n< state = {item[2][0]}, action = {action} g(n) = {updated_cost}, d = {len(item[2])}, f(n) = {updated_cost + updated_weight}, Parent = Pointer to {updated_state}>")
                file.write("\n]\n")
     # Prepare the steps for the solution path
    step = []
    for i in range(len(path)):
        step.append([path[i], move_weight[i]])

    return popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost, step

# Depth Limited Search

# Function to perform Depth-Limited Search (DLS) to find the goal state
def dls_search(start, goal, limit):
    # StateNodeDL class to represent a state in the search tree
    class StateNodeDL:
        def __init__(self, state, parent, move, cost, weight):
            self.state = state #similar to previous ones
            self.parent = parent
            self.move = move
            self.cost = cost
            self.weight = weight

# Function to fetch possible moves based on the position of the blank tile
    def fetch_moves(state):
        moves = []

        blank_pos = state.index(0)
        if blank_pos % 3 > 0:
            moves.append(('Left', state[blank_pos-1]))
        if blank_pos % 3 < 2:
            moves.append(('Right', state[blank_pos+1]))
        if blank_pos // 3 > 0:
            moves.append(('Up', state[blank_pos-3]))
        if blank_pos // 3 < 2:
            moves.append(('Down', state[blank_pos+3]))
        return moves

    visited_states = set() #similar to previous ones
    initial_node = StateNodeDL(start, None, None, 0, 0)
    stack = [(initial_node, 0)]
    popped_count = 0
    max_stack_size = 0
    expanded_count = 0

    move_offset = {'Left': -1, 'Right': 1, 'Up': -3, 'Down': 3} # Offsets for moves

    if dump == 'true':
        file = open(timestamp, 'a')

    while stack:
        if len(stack) > max_stack_size:
            max_stack_size = len(stack)  # Update the maximum stack size
        state_node, depth = stack.pop() # Get the next node from the stack
        popped_count += 1 # Increment the popped count
        if dump == 'true':
            file.write(f"Generating successors to < state = {state_node.state}, action = {state_node.move} g(n) = {state_node.cost}, d = {depth}, f(n) = {state_node.cost + depth}, Parent = Pointer to {None} >:\n")

        # Check if the current state is the goal state
        if state_node.state == goal:
            path = []
            total_cost = state_node.cost
            
             # Reconstruct the path to the goal
            while state_node.parent is not None:
                if (state_node.move == "Down"):
                    state_node.move = "Up"
                elif (state_node.move == "Left"):
                    state_node.move = "Right"
                elif (state_node.move == "Up"):
                    state_node.move = "Down"

                else:
                    state_node.move = "Left"
                path.append((state_node.move, state_node.weight))
                state_node = state_node.parent
            path.reverse()
            return popped_count, expanded_count, len(visited_states), max_stack_size, len(path), total_cost, path

        # Expand the current node if the depth is within the limit
        if depth < limit:
            visited_states.add(tuple(state_node.state))
            expanded_count += 1 # Increment the expanded count

            # Generate successors for the current node
            for move, next_cost in fetch_moves(state_node.state):
                child_state = state_node.state.copy()
                blank_pos = child_state.index(0)
                child_state[blank_pos] = next_cost
                child_state[blank_pos + move_offset[move]] = 0
                weight = next_cost
                if tuple(child_state) not in visited_states:
                    child_node = StateNodeDL(child_state, state_node, move, state_node.cost + next_cost, weight)
                    stack.append((child_node, depth+1))  # Add the child node to the stack
                    if dump == 'true':
                        file.write(f"< state = {child_node.state}, action = {child_node.move} g(n) = {child_node.cost}, d = {depth+1}, f(n) = {child_node.cost + depth + 1}, Parent = Pointer to {child_node.parent.state} >\n")

    return None

# Function to perform Iterative Deepening Depth-First Search (IDDFS)
def dls_solution(start, goal, depth_limit):
    result = None
    # Iteratively increase the depth limit and perform DLS
    for i in range(depth_limit, sys.maxsize):
        result = dls_search(start, goal, i)
        if result is not None:
            break
    return result

'''Iterative Deepening Search'''

# Function to perform Iterative Deepening Search (IDS) to find the goal state
def ids_search(start, goal):
    depth_limit = 0  # Initialize the depth limit to 0

    # Iteratively increase the depth limit and perform Depth-Limited Search (DLS)
    while True:
        result = dls_solution(start, goal, depth_limit) # Perform DLS with the current depth limit
        
        # If a solution is found then return the result
        if result is not None:
            return result
        
         # Increase the depth limit for the next iteration
        depth_limit += 1


'''Greedy Search'''
# StateNodeH class to represent a state in the search tree
class StateNodeH:
    def __init__(self, state, parent, action, cost, heuristic, weight):
        self.state = state #similar to previous codes
        self.cost = cost
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.weight = weight
        self.total_cost = self.cost + self.heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost # Compare nodes based on total cost

# Function to generate a new state based on the given action
def generate_new_state_heuristic(state, action):
    updated_state = [row[:] for row in state]
    blank_i, blank_j = locate_blank_position(updated_state)
    action_offset = {
        'Up': (-1, 0),
        'Down': (1, 0),
        'Left': (0, -1),
        'Right': (0, 1)
    }
    offset_i, offset_j = action_offset[action]
    updated_state[blank_i][blank_j], updated_state[blank_i+offset_i][blank_j+offset_j] = \
        updated_state[blank_i+offset_i][blank_j+offset_j], updated_state[blank_i][blank_j]
    return updated_state

# Function to calculate the Manhattan distance between two points
def calculate_manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
# Function to compute the heuristic value of a state using the Manhattan distance
def compute_heuristic(state, goal):
    def calculate_manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    heuristic = 0
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val != goal[i][j]:
                goalpos = next((m, n) for m, row in enumerate(goal) for n, v in enumerate(row) if v == val)
                heuristic += calculate_manhattan_distance((i, j), goalpos)
    return heuristic

# Function to fetch possible actions based on the position of the blank tile
def fetch_actions_heuristic(state):
    blank_i, blank_j = locate_blank_position(state)
    row, col = len(state), len(state[0])
    return [action for action, (i, j) in [('Up', (blank_i-1, blank_j)), ('Down', (blank_i+1, blank_j)),
                                          ('Left', (blank_i, blank_j-1)), ('Right', (blank_i, blank_j+1))]
            if 0 <= i < row and 0 <= j < col]

# Function to retrieve the path from the start state to the goal state
def retrieve_path(state_node):
    path = []
    while state_node:
        if state_node.action:
            path.append((state_node.action, state_node.weight))
        state_node = state_node.parent
    return list(reversed(path))

# Function to locate the position of the blank tile (0) in the puzzle
def locate_blank_position(state):
    for i, row in enumerate(state):
        if 0 in row:
            return (i, row.index(0))

# Function to perform Greedy Search to find the goal state
def greedy_search_algorithm(start, goal):
    popped_count = 0 # similar to before
    expanded_count = 0
    max_queue_size = 0
    generated_count = 1
    search_depth = 0
    total_cost = 0

    # Create root node
    initial_node = StateNodeH(start, None, None, 0, compute_heuristic(start, goal), None)

    # Initialize fringe and visited set
    priority_queue = PriorityQueue()
    visited_states = set()

    # Add root node to fringe
    priority_queue.put(initial_node)

    # Log initial state of fringe and visited set
    if dump == 'true':
        file = open(timestamp, 'a')
        file.write(f"Generating successors to < state = {initial_node.state}, action = {{Start}} g(n) = {initial_node.cost}, d = 0, f(n) = {initial_node.heuristic}, Parent = Pointer to {{None}} >:\n")
        file.write(f"Closed: {visited_states}\n")
        file.write(f"Fringe: [{initial_node}]\n")

    while not priority_queue.empty():
        # Update max queue or fringe size
        max_queue_size = max(max_queue_size, priority_queue.qsize())

        # Pop node from fringe
        state_node = priority_queue.get()
        popped_count += 1

        # Check if goal state is reached
        if state_node.state == goal:
            # Update depth, cost, and steps
            search_depth = len(retrieve_path(state_node))
            total_cost = state_node.cost
            path = retrieve_path(state_node)
            return popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost, path

        # Add current node to visited set
        visited_states.add(str(state_node.state))

        # Log state of current node and visited set
        if dump == 'true':
            file.write(f"Generating successors to < state = {state_node.state}:\n")
            file.write(f"\tClosed: {visited_states}\n")

        # Generate and add successors to fringe
        for action in fetch_actions_heuristic(state_node.state):
            # Generate new state based on action
            updated_state = generate_new_state_heuristic(state_node.state, action)

            # Calculate new cost
            updated_cost = state_node.cost + updated_state[locate_blank_position(state_node.state)[0]][locate_blank_position(state_node.state)[1]]

            # Check if new state has already been visited
            if str(updated_state) not in visited_states:
                # Invert action for logging purposes
                if action == "Down":
                    action = "Up"
                elif action == "Up":
                    action = "Down"
                elif action == "Left":
                    action = "Right"
                else:
                    action = "Left"

                # Create new node and add to the priority queue
                weight = updated_state[locate_blank_position(state_node.state)[0]][locate_blank_position(state_node.state)[1]]
                new_node = StateNodeH(updated_state, state_node, action, updated_cost, compute_heuristic(updated_state, goal), weight)
                if new_node.heuristic <= 70:
                    generated_count += 1
                    priority_queue.put(new_node)

                    # Log state of new node and fringe
                    if dump == 'true':
                        file.write(f"\tFringe: {new_node.state}\n")

        # Update nodes expanded
        expanded_count += 1

    return None

# Function to perform A* Search to find the goal state
def a_star_algorithm(start, goal):
    # Initialize variables
    popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost = 0, 0, 1, 0, 0, 0  #similar to before
    path, visited_states = [], set()
    priority_queue = PriorityQueue()

    # Add start node to the fringe- priority queue
    initial_node = StateNodeH(start, None, None, 0, compute_heuristic(start, goal), None)
    priority_queue.put(initial_node)

    # If dump is true, create a log file and write the first entry
    if dump:
        with open(timestamp, 'a') as file:
            file.write(f"Generating successors to < state = {initial_node.state}, action = {{Start}} g(n) = {initial_node.cost}, d = 0, f(n) = {initial_node.heuristic}, Parent = Pointer to {{None}} >:\n")
            file.write(f"\tClosed: {visited_states}\n")
            file.write(f"\tFringe: [{initial_node}]\n")

    # While the fringe is not empty, keep searching
    while not priority_queue.empty():
        # Update the maximum fringe size seen so far
        max_queue_size = max(max_queue_size, priority_queue.qsize())

        # Get the next node from the fringe and increment the nodes popped counter
        state_node = priority_queue.get()
        popped_count += 1

        # If the node is the goal state, construct the path and return results
        if state_node.state == goal:
            search_depth = len(retrieve_path(state_node))
            total_cost = state_node.cost
            path = retrieve_path(state_node)
            return popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost, path

        # Add the node to the set of visited states
        visited_states.add(tuple(map(tuple, state_node.state)))

        # If dump is true, write a log entry for generating successors for the current node
        if dump:
            with open(timestamp, 'a') as file:
                file.write(f"Generating successors to < state = {state_node.state}:\n")

        # Generate successors for the current node
        for action in fetch_actions_heuristic(state_node.state):
            updated_state = generate_new_state_heuristic(state_node.state, action)
            updated_cost = state_node.cost + updated_state[locate_blank_position(state_node.state)[0]][locate_blank_position(state_node.state)[1]]

            # If the new state has not been visited before, create a new node for it and add it to the fringe
            if tuple(map(tuple, updated_state)) not in visited_states:
                if action == "Down":
                    action = "Up"
                elif action == "Up":
                    action = "Down"
                elif action == "Left":
                    action = "Right"
                else:
                    action = "Left"
                generated_count += 1
                weight = updated_state[locate_blank_position(state_node.state)[0]][locate_blank_position(state_node.state)[1]]
                new_node = StateNodeH(updated_state, state_node, action, updated_cost, compute_heuristic(updated_state, goal), weight)

                # If the new node's heuristic value is not too high, add it to the fringe
                if new_node.heuristic <= 50:
                    if dump:
                        with open(timestamp, 'a') as file:
                            file.write(f"Closed: {visited_states}\n")
                            file.write(f"Fringe: {new_node.state}\n")
                    generated_count += 1
                    priority_queue.put(new_node)

        # Increment the nodes expanded counter
        expanded_count += 1

    return "No solution found."

def change_list(data):
    return [data[i:i+3] for i in range(0, 9, 3)]

if __name__ == '__main__':
    if len(sys.argv) != 5:

        sys.exit()
    method = 'a*'
    dump = 'false'
    initial_file = sys.argv[1]
    target_file = sys.argv[2]

    now = datetime.datetime.now()
    timestamp = now.strftime("trace-%m_%d_%Y-%I_%M_%S_%p.txt")

    if (len(sys.argv) == 4):
        if sys.argv[3] == 'true' or sys.argv[3] == 'false':
            dump = sys.argv[3]
        else:
            method = sys.argv[3]
    if len(sys.argv) == 5:
        method = sys.argv[3]
        dump = sys.argv[4]

    
    initial_file_handle = open(initial_file, "r")
    target_file_handle = open(target_file, "r")

    initial_state = []
    target_state = []

    with open(initial_file, 'r') as initial_file_handle:
        for i in initial_file_handle:
            if i == 'END OF FILE':
                break
            i = i.replace("\n", '')
            i = i.split(" ")
            for j in i:
                initial_state.append(int(j))

    with open(target_file, 'r') as target_file_handle:
        for i in target_file_handle:
            if i == 'END OF FILE':
                break
            i = i.replace("\n", '')
            i = i.split(" ")
            for j in i:
                target_state.append(int(j))

    if dump == 'true':
        file = open(timestamp, 'a')
        file.write(f"Command-Line Arguments : ['{initial_file}', '{target_file}', '{method}', '{dump}']\n")
        file.write(f"Method selection: {method}\n")
        file.write(f"Running {method}\n")
        file.close()

    if (method == "bfs"):
        search_result = bfs_search(tuple(initial_state), tuple(target_state))

    elif (method == "dfs"):
        search_result = dfs_search(initial_state, target_state)
    elif (method == "ucs"):
        initial_state = change_list(initial_state)
        target_state = change_list(target_state)
        search_result = ucs_search(initial_state, target_state)
    elif (method == "dls"):
        depth_limit = int(input("Enter depth limit: "))
        search_result = dls_solution(initial_state, target_state, depth_limit)

    elif (method == "greedy"):
        initial_state = change_list(initial_state)
        target_state = change_list(target_state)
        search_result = greedy_search_algorithm(initial_state, target_state)
    elif (method == "ids"):
        search_result = ids_search(initial_state, target_state)
    else:
        initial_state = change_list(initial_state)
        target_state = change_list(target_state)
        search_result = a_star_algorithm(initial_state, target_state)

    popped_count, expanded_count, generated_count, max_queue_size, search_depth, total_cost, path = search_result

    print("Nodes Popped: {}".format(popped_count))
    print("Nodes Expanded: {}".format(expanded_count))
    print("Nodes Generated:{}".format(generated_count))
    print("Max Fringe Size: {}".format(max_queue_size))
    print("Solution Found at depth", len(path), "with cost of {}.".format(total_cost))
    print("Steps:")
    for move in path:
        print("\tMove", move[1], move[0])

    if dump == 'true':
        file = open(timestamp, 'a')
        file.write(f"\tNodes Popped: {popped_count}\n")
        file.write(f"\tNodes expanded: {expanded_count}\n")
        file.write(f"\tNodes Generated: {generated_count}\n")
        file.write(f"\tMax Fringe Size: {max_queue_size}\n")



