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

    for neighbour in graph[m]:<br>
      if neighbour not in visited:<br>
        visited.append(neighbour)<br>
        queue.append(neighbour)<br>

# Driver Code<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')    # function calling<br>

output:<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7 <br>
