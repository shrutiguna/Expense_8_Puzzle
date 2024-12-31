# **Expense 8 Puzzle Problem**

## **Overview**
The Expense 8 Puzzle is a modified version of the classic 8 Puzzle problem. It involves solving a 3x3 grid where 8 numbered tiles and one blank tile can be moved to achieve a target configuration. The modification adds a cost element: moving a tile incurs a cost equal to the number on the tile. The task is to determine the sequence of moves that solves the puzzle with the lowest possible cost using various search algorithms.

This implementation includes algorithms such as Breadth-First Search (BFS), Uniform Cost Search (UCS), Depth-First Search (DFS), Depth-Limited Search (DLS), Iterative Deepening Search (IDS), Greedy Search, and A* Search. The default algorithm is A* if no method is specified.

How the code is structured: 
- This folder contains expense_8_puzzle.py, start.txt, goal.txt
- start.txt has the initial start state of the puzzle
- goal.txt has the final state of the puzzle
- dump files are also created in the same folder when the code is executed. The dump file will contain- expanded nodes, the fringe, and the steps taken to reach the goal.

How to run the file:

1) Open command prompt or windows powershell.
2) Navigate to the folder containing the .py file, start.txt and goal.txt using the "cd" command.
3) - In the shell type-Syntax : python <filename.py> <startfile> <endfile> <algorithm> <dumpflag>
     Example- python expense_8_puzzle.py start.txt goal.txt a* true" to run the code for a* algorithm and for dump file creation 
   - Syntax : python <filename.py> <startfile> <endfile> <algorithm> <dumpflag>
   - make sure to enter the correct start and end file
4) Similarly to run other algorithms, instead of a*, put in bfs or dfs or ucs or dls or greedy or ids to run the algorithm for it. You can enter false for the dump flag too if you do not want to create the dump file. 
5) Dump flag only has 2 parameters- true or false- based on which it will generate.
6) For dls, after executing the command, it will ask you to enter the depth limit to search on.

