# AIL
1. Write a Program to Implement Breadth First Search using Python.<br>
graph = {<br>
 '1' : ['2','10'],<br>
 '2' : ['3','8'],<br>
 '3' : ['4'],<br>
 '4' : ['5','6','7'],<br>
 '5' : [],<br>
 '6' : [],<br>
 '7' : [],<br>
 '8' : ['9'],<br>
 '9' : [],<br>
 '10' : []<br>
 }<br>


visited = [] # List for visited nodes.<br>
queue = []     #Initialize a queue<br>

def bfs(visited, graph, node): #function for BFS<br>
  visited.append(node)<br>
  queue.append(node)<br>

  while queue:          # Creating loop to visit each node<br>
    m = queue.pop(0) <br>
    print (m, end = " ") <br>

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

#Driver Code<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')    # function calling<br>

output:<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7 <br>

2.  Write a Program to Implement Depth First Search using Python.<br>
#Using a Python dictionary to act as an adjacency list<br>
graph = {<br>
 '5' : ['3','7'],<br>
 '3' : ['2', '4'],<br>
 '7' : ['6'],<br>
 '6': [],<br>
 '2' : ['1'],<br>
 '1':[],<br>
 '4' : ['8'],<br>
 '8' : []<br>
}<br>
visited = set() # Set to keep track of visited nodes of graph.<br>

def dfs(visited, graph, node):  #function for dfs <br>
    if node not in visited:<br>
        print (node)<br>
        visited.add(node)<br>
        for neighbour in graph[node]:<br>
            dfs(visited, graph, neighbour)<br>

#Driver Code<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>

3.write a program to mpement breadth first search.<br>
from queue import PriorityQueue.<br>
import matplotlib.pyplot as plt.<br>
import networkx as nx.<br>
#for implementing BFS | returns path having lowest cost.<br>
def best_first_search(source, target, n):.<br>
 visited = [0] * n.<br>
 visited[source] = True.<br>
 pq = PriorityQueue().<br>
 pq.put((0, source)).<br>
 while pq.empty() == False:.<br>
   u = pq.get()[1].<br>
   print(u, end=" ") # the path having lowest cost.<br>
   if u == target:.<br>
     break.<br>
   for v, c in graph[u]:.<br>
     if visited[v] == False:.<br>
       visited[v] = True.<br>
       pq.put((c, v)).<br>
       print().<br>
#for adding edges to graph.<br>
def addedge(x, y, cost):<br>
 graph[x].append((y, cost))<br>
 graph[y].append((x, cost))<br>

v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
 x, y, z = list(map(int, input().split()))<br>
 addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("Path: ", end = "")<br>
best_first_search(source, target, v)<br>

output:<br>
Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1  1<br>
0 2 1<br>
0 3 2<br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br>
Enter the Target/Destination Node: 1<br>
Path: 2 <br>

0 <br>
1 <br>

4.write a program to implement waterjug problem using python <br>
from collections import defaultdict <br>
jug1, jug2, aim = 4, 3, 2 <br>
visited = defaultdict(lambda: False) <br>
def waterJugSolver(amt1, amt2): <br>
 if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0): <br>
   print(amt1, amt2) <br>
   return True <br>
 if visited[(amt1, amt2)] == False: <br>
   print(amt1, amt2) <br>
   visited[(amt1, amt2)] = True <br>
   return (waterJugSolver(0, amt2) or <br>
 waterJugSolver(amt1, 0) or <br>
 waterJugSolver(jug1, amt2) or <br>
 waterJugSolver(amt1, jug2) or <br>
 waterJugSolver(amt1 + min(amt2, (jug1-amt1)), <br>
 amt2 - min(amt2, (jug1-amt1))) or <br>
 waterJugSolver(amt1 - min(amt1, (jug2-amt2)), <br>
 amt2 + min(amt1, (jug2-amt2)))) <br>
 else: <br>
   return False <br>
print("Steps: ") <br>
waterJugSolver(0, 0) <br>

output: <br>
Steps:  <br>
0 0 <br>
4 0 <br>
4 3 <br>
0 3 <br>
3 0 <br>
3 3 <br>
4 2 <br>
0 2 <br>
True <br>
5. Write a Program to Implement Tower of Hanoi using Python. <br>
def TowerOfHanoi(n , source, destination, auxiliary): <br>
    if n==1: <br>
        print ("Move disk 1 from source",source,"to destination",destination) <br>
        return <br>
    TowerOfHanoi(n-1, source, auxiliary, destination) <br>
    print ("Move disk",n,"from source",source,"to destination",destination) <br>
    TowerOfHanoi(n-1, auxiliary, destination, source) <br>

n = 3 <br>
TowerOfHanoi(n,'A','B','C') <br>

output:
Move disk 1 from source A to destination B <br>
Move disk 2 from source A to destination C <br>
Move disk 1 from source B to destination C <br>
Move disk 3 from source A to destination B <br>
Move disk 1 from source C to destination A <br>
Move disk 2 from source C to destination B <br>
Move disk 1 from source A to destination B <br>

6.write a python program to implement tic-toc-toe. <br>
#Tic-Tac-Toe Program using <br>
#random number in Python <br>
#importing all necessary libraries <br>
import numpy as np <br>
import random <br>
from time import sleep <br>
 
#Creates an empty board <br>
def create_board(): <br>
    return(np.array([[0, 0, 0], <br>
                     [0, 0, 0], <br>
                     [0, 0, 0]])) <br>
 
#Check for empty places on board <br>
 def possibilities(board): <br>
    l = [] <br>
 
    for i in range(len(board)): <br>
        for j in range(len(board)): <br>
 
            if board[i][j] == 0: <br>
                l.append((i, j)) <br>
    return(l) <br>
 
#Select a random place for the player <br>
def random_place(board, player): <br>
    selection = possibilities(board) <br>
    current_loc = random.choice(selection) <br>
    board[current_loc] = player <br>
    return(board) <br>
 
#Checks whether the player has three <br>
#of their marks in a horizontal row  <br>
def row_win(board, player): <br>
    for x in range(len(board)): <br>
        win = True <br>
 
        for y in range(len(board)): <br>
            if board[x, y] != player: <br>
                win = False <br>
                continue <br>
 
        if win == True: <br>
            return(win) <br>
    return(win) <br>
 
#Checks whether the player has three <br>
#of their marks in a vertical row <br>
def col_win(board, player): <br>
    for x in range(len(board)): <br>
        win = True <br>
 
        for y in range(len(board)): <br>
            if board[y][x] != player: <br>
                win = False
                continue <br>
 
        if win == True: <br>
            return(win) <br>
    return(win) <br>
 
#Checks whether the player has three <br>
#of their marks in a diagonal row  <br>
def diag_win(board, player): <br>
    win = True <br>
    y = 0 <br>
    for x in range(len(board)): <br>
        if board[x, x] != player: <br>
            win = False <br>
    if win: <br>
        return win <br>
    win = True <br>
    if win: <br>
        for x in range(len(board)): <br>
            y = len(board) - 1 - x <br>
            if board[x, y] != player: <br>
                win = False <br>
    return win  <br>
#Evaluates whether there is <br>
#inner or a tie <br>
def evaluate(board): <br>
    winner = 0 <br>
 
    for player in [1, 2]: <br>
        if (row_win(board, player) or <br>
                col_win(board, player) or <br>
                diag_win(board, player)): <br>
 
            winner = player <br>
 
    if np.all(board != 0) and winner == 0: <br>
        winner = -1
    return winner <br>
 
#Main function to start the game <br>
def play_game(): <br>
    board, winner, counter = create_board(), 0, 1 <br>
    print(board) <br>
    sleep(2) <br>
 
    while winner == 0: <br>
        for player in [1, 2]: <br>
            board = random_place(board, player) <br>
            print("Board after " + str(counter) + " move") <br>
            print(board) <br>
            sleep(2) <br>
            counter += 1 <br>
            winner = evaluate(board) <br>
            if winner != 0: <br>
                break <br>
    return(winner) <br>
#Driver Code <br>
print("Winner is: " + str(play_game())) <br>

output:
![image](https://user-images.githubusercontent.com/97970956/208869442-ba12fcbf-3940-400a-b6b5-f4c38ce32718.png)<br>
![image](https://user-images.githubusercontent.com/97970956/208869644-d412b24e-7f04-4b7b-9dc4-70b1fe50bf49.png)<br>


7.travelling salesman problem <br>
from sys import maxsize <br>
from itertools import permutations <br>
V=4 <br>
def travellingSalesmanProblem(graph,s): <br>
    #store all vertex apart from source vertex <br>
    vertex=[] <br>
    for i in range(V): <br>
        if i!=s: <br>
            vertex.append(i) <br>
    #store minimum weight Hamiltonian Cycle   <br>  
    min_path=maxsize <br>
    next_permutation=permutations(vertex) <br> <br>
    for i in next_permutation: <br>
        #store current Path weight(cost) <br>
        current_pathweight=0 <br>
        #compute current path weight <br>
        k=s <br>
        for j in i: <br>
            current_pathweight+=graph[k][j] <br>
            k=j <br>
        current_pathweight+=graph[k][s] <br>
        #Update minimum <br>
        min_path=min(min_path,current_pathweight) <br>
    return min_path <br>
#Driver Code <br>
if __name__ =="__main__": <br>
    
       #matrix representation of graph <br>
    graph=[[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]] <br>
    s=0 <br>
    print(travellingSalesmanProblem(graph,s)) <br>
 
 
 output:<br>
 ![image](https://user-images.githubusercontent.com/97970956/209785651-705557f1-8fed-4b0c-b9ec-b48d1cf2e3b8.png)<br>
 
 8.wite a program to implement the n quuen problem<br>
 global N<br>
N = 4<br>
def printSolution(board):<br>
 for i in range(N):<br>
   for j in range(N):<br>
     print (board[i][j], end = " ")<br>
   print()<br>
def isSafe(board, row, col):<br>
 for i in range(col):<br>
   if board[row][i] == 1:<br>
    return False<br>
 for i, j in zip(range(row, -1, -1),range(col, -1, -1)):<br>
   if board[i][j] == 1:<br>
    return False<br>
 for i, j in zip(range(row, N, 1),range(col, -1, -1)):<br>
   if board[i][j] == 1:<br>
    return False<br>
 return True<br>
def solveNQUtil(board, col):<br>
 if col >= N:<br>
   return True<br>
 for i in range(N):<br>
   if isSafe(board, i, col):<br>
     board[i][col] = 1 <br>
     if solveNQUtil(board, col + 1) == True:<br>
       return True<br>
     board[i][col] = 0<br>
 return False<br>
def solveNQ():<br>
 board = [ [0, 0, 0, 0],<br>
 [0, 0, 0, 0],<br>
 [0, 0, 0, 0],<br>
 [0, 0, 0, 0] ]<br>
 if solveNQUtil(board, 0) == False:<br>
   print ("Solution does not exist")<br>
   return False<br>
 printSolution(board)<br>
 return True<br>
solveNQ()<br>

output:<br>
![image](https://user-images.githubusercontent.com/97970956/210524319-8fd42452-6261-40fc-b459-c27058aa0004.png)<br>

9.write a program to implement the 8 puzzel problem<br>
#8 puzzel problem<br>
import copy<br>
from heapq import heappush, heappop<br>
n = 3<br>
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>
class priorityQueue:<br>
    def __init__(self):<br>
        self.heap = []<br>
    def push(self, k):<br>
        heappush(self.heap, k)<br>
    def pop(self):<br>
        return heappop(self.heap)<br>
    def empty(self):<br>
        if not self.heap:<br>
            return True<br>
        else:<br>
            return False<br>
class node:<br>
     
    def __init__(self, parent, mat, empty_tile_pos,<br>
                 cost, level):<br>
        self.parent = parent<br>
        self.mat = mat<br>
        self.empty_tile_pos = empty_tile_pos<br>
        self.cost = cost<br>
        self.level = level<br>
    def __lt__(self, nxt):<br>
        return self.cost < nxt.cost<br>
def calculateCost(mat, final) -> int: <br>    
    count = 0<br>
    for i in range(n):<br>
        for j in range(n):<br>
            if ((mat[i][j]) and<br>
                (mat[i][j] != final[i][j])):<br>
                count += 1  <br>               
    return count <br>
def newNode(mat, empty_tile_pos, new_empty_tile_pos,<br>
            level, parent, final) -> node:<br>
    new_mat = copy.deepcopy(mat) <br>
    x1 = empty_tile_pos[0]<br>
    y1 = empty_tile_pos[1]<br>
    x2 = new_empty_tile_pos[0]<br>
    y2 = new_empty_tile_pos[1]<br>
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>
    cost = calculateCost(new_mat, final)<br>
    new_node = node(parent, new_mat, new_empty_tile_pos,<br>
                    cost, level)<br>
    return new_node<br>
def printMatrix(mat):<br>
    for i in range(n):<br>
        for j in range(n):<br>
            print("%d " % (mat[i][j]), end = " ") <br>            
        print()<br>
def isSafe(x, y): <br>    
    return x >= 0 and x < n and y >= 0 and y < n<br>
def printPath(root): <br>    
    if root == None:<br>
        return     <br><br>
    printPath(root.parent)<br>
    printMatrix(root.mat)<br>
    print()<br>
def solve(initial, empty_tile_pos, final):<br>
    pq = priorityQueue()<br>
    cost = calculateCost(initial, final)<br>
    root = node(None, initial,<br>
                empty_tile_pos, cost, 0)<br>
    pq.push(root)<br>
    while not pq.empty():<br>
        minimum = pq.pop()<br>
        if minimum.cost == 0:<br>
            printPath(minimum)<br>
            return<br>
        for i in range(n):<br>
            new_tile_pos = [<br>
                minimum.empty_tile_pos[0] + row[i],<br>
                minimum.empty_tile_pos[1] + col[i], ]  <br>               
            if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
                child = newNode(minimum.mat,<br>
                                minimum.empty_tile_pos,<br>
                                new_tile_pos,<br>
                                minimum.level + 1,<br>
                                minimum, final,)<br>
                pq.push(child)<br>
initial = [ [ 1, 2, 3 ],<br>
            [ 5, 6, 0 ],<br>
            [ 7, 8, 4 ] ]<br>
final = [ [ 1, 2, 3 ],<br>
          [ 5, 8, 6 ],<br>
          [ 0, 7, 4 ] ]<br>
empty_tile_pos = [ 1, 2 ]<br>
solve(initial, empty_tile_pos, final)<br>

output:<br>
![image](https://user-images.githubusercontent.com/97970956/210524797-d02582aa-aff2-41b1-bb03-2c81bc0b58ca.png)<br>

10.A* algorithm<br>
#a star algorithm<br>
def aStarAlgo(start_node, stop_node):<br>
         
        open_set = set(start_node) <br>
        closed_set = set()<br>
        g = {} #store distance from starting node<br>
        parents = {}# parents contains an adjacency map of all nodes<br>
 
        #ditance of starting node from itself is zero<br>
        g[start_node] = 0<br>
        #start_node is root node i.e it has no parent nodes<br>
        #so start_node is set to its own parent node<br>
        parents[start_node] = start_node<br>
         
         
        while len(open_set) > 0:<br>
            n = None<br>
 
            #node with lowest f() is found<br>
            for v in open_set:<br>
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):<br>
                    n = v<br>
             
                     
            if n == stop_node or Graph_nodes[n] == None:<br>
                pass<br>
            else:<br>
                for (m, weight) in get_neighbors(n):<br>
                    #nodes 'm' not in first and last set are added to first<br>
                    #n is set its parent<br>
                    if m not in open_set and m not in closed_set:<br>
                        open_set.add(m)<br>
                        parents[m] = n<br>
                        g[m] = g[n] + weight<br>
                         
     
                    #for each node m,compare its distance from start i.e g(m) to the<br>
                    #from start through n node<br>
                    else:<br>
                        if g[m] > g[n] + weight:<br>
                            #update g(m)<br>
                            g[m] = g[n] + weight<br>
                            #change parent of m to n<br>
                            parents[m] = n<br>
                             
                            #if m in closed set,remove and add to open<br>
                            if m in closed_set:<br>
                                closed_set.remove(m)<br>
                                open_set.add(m)<br>
 
            if n == None:<br>
                print('Path does not exist!')<br>
                return None<br>
 
            # if the current node is the stop_node<br>
            # then we begin reconstructin the path from it to the start_nod<br>
            if n == stop_node:<br>
                path = []<br>
 
                while parents[n] != n:<br>
                    path.append(n)<br>
                    n = parents[n]<br>
 
                path.append(start_node)<br>
 
                path.reverse()<br>
 
                print('Path found: {}'.format(path))<br>
                return path<br>
 
 
            # remove n from the open_list, and add it to closed_list<br>
            # because all of his neighbors were inspected<br>
            open_set.remove(n)<br>
            closed_set.add(n)<br>
 
        print('Path does not exist!')<br>
        return None<br>
         
#define fuction to return neighbor and its distance<br>
#from the passed node<br>
def get_neighbors(v):<br>
    if v in Graph_nodes:<br>
        return Graph_nodes[v]<br>
    else:<br>
        return None<br>
#for simplicity we ll consider heuristic distances given<br>
#and this function returns heuristic distance for all nodes<br>
def heuristic(n):<br>
        H_dist = {<br>
            'A': 11,<br>
            'B': 6,<br>
            'C': 99,<br>
            'D': 1,<br>
            'E': 7,<br>
            'G': 0,<br>
             
        }<br>
 
        return H_dist[n]<br>
 
#Describe your graph here  <br>
Graph_nodes = {<br>
    'A': [('B', 2), ('E', 3)],<br>
    'B': [('C', 1),('G', 9)],<br>
    'C': None,<br>
    'E': [('D', 6)],<br>
    'D': [('G', 1)],<br>
     
}<br>
aStarAlgo('A', 'G')<br>

output:<br>
![image](https://user-images.githubusercontent.com/97970956/210525773-c8a4c0ee-da96-4a93-a128-029d0adaa7ab.png)<br>
