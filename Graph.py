import sys
import copy


# Min_Binary_Heap class
class Heap():

    # Converts the graph to min heap
    def min_heapify(A, k):
        left = 2 * k + 1
        right = 2 * k + 2
        smallest = left if left < len(A) and A[left] < A[k] else k
        if right < len(A) and A[right] < A[smallest]:
            smallest = right
        if smallest != k:
            A[k], A[smallest] = A[smallest], A[k]
            min_heapify(A, smallest)

    def build_min_heap(A):
        m = int((len(A)//2)-1)
        for k in range(m, -1, -1):
            min_heapify(A, k)


    def heappop(heap):
        first_element = heap[0]
        heap[0] = heap[-1]
        del heap[-1]
        min_heapify(heap, 0)
        return first_element


    def heappush(self, A, element):
        A.append(element)


#  Represents a vertex in the graph.
class Vertex:
    def __init__(self, nm):
        self.name = nm  # Vertex name
        self.adj = []  # Adjacent vertices, [[vertex:Vertex, cost], [], ....]

    def reset(self):
        self.prev = None

    def __gt__(self, other):
        return self.name > other.name

    def __cmp__(self, other):
        return cmp(self.name, other.name)

# Graph Class
class Graph:
    def __init__(self):
        self.vertexMap = {}  # vertexName:string -> vertex:Vertex
        self.vertexDown = {}  # vertex:Vertex -> cost:float
        # vertex:Vertex -> [[vertex:Vertex, cost:float], [], ....]
        self.edgeDown = {}

    # Opening the file to store the output
    def setFile(self, fileName):
        self.file = open(fileName, "w")
    
    # Closing the file and freeing all the resources assigned 
    def closeFile(self):
        self.file.close()

    # Add a new edge to the graph.
    def addEdge(self, sourceName,  destName, cost):
        v = self.getVertex(sourceName)
        w = self.getVertex(destName)
        for edge in v.adj:
            if edge[0] == w:
                edge[1] = min(edge[1], cost)
                return
        v.adj.append([w, cost])

    # If vertexName is not present, add it to vertexMap.
    # In either case, return the Vertex.
    def getVertex(self, vertexName):
        if vertexName not in self.vertexMap:
            v = Vertex(vertexName)
            self.vertexMap[vertexName] = v
        return self.vertexMap[vertexName]

    # Checks whether an edge is down or not
    def containsEdgeDown(self, fromVertex: Vertex, toVertex: Vertex):
        if fromVertex in self.edgeDown:
            adj = [edge[0] for edge in self.edgeDown[fromVertex]]
            return toVertex in adj
        return False

    # Finding the shortest path using Dijkstra's Algorithm using Minimum Binary Heap Implementation
    def calculatePath(self, startName, destName):
        start = self.getVertex(startName)
        dest = self.getVertex(destName)
        heap = [[0, [start]]]   # [[cost till now, [vertex:Vertex, vertex:Vertex, ....]], [cost till now, [...]]]
        dist = {}   # vertex:Vertex -> [cost of best path, [vertex:Vertex, vertex:Vertex, ....]]
        
        # Run time O(V+E)
        while heap:
            top = heappop(heap)
            for edge in top[1][-1].adj:
                checkDown = self.containsEdgeDown(
                    top[1][-1], edge[0]) or top[1][-1] in self.vertexDown or edge[0] in self.vertexDown
                checkLow = edge[0] not in dist or top[0] + \
                    edge[1] < dist[edge[0]][0]
                if checkDown == False and checkLow:
                    newPath = copy.deepcopy(top[1])
                    newPath.append(edge[0])
                    dist[edge[0]] = [top[0] + edge[1], newPath]
                    heappush(heap, dist[edge[0]])
        return dist[dest]

    # Function to Perform Reachable command
    def printReachable(self):

        """"" This function identifies and returns the list of all the reachable vertices that can be reached by valid path for each vertex.
        The algorithm that I have used to identify the set of vertices that are reachable from each vertex by valid path is as follows,
        First line(line 124) retrieves all the vertices in the graph and sorts them in the ascending order.
        Then a for loops iterates over each vertex in the sorted list of vertices.
        Then it checks if the current vertex is in down state or up state. And if it's not in down state the we are writing the vertex name to a output file.
        And then to store all the reachable vertices from current vertex I created and empty list called 'edges'.
        Then a queue 'q' is initialized with current vertex 'v' and a dictionary 'vis' is initialized with 'v' as the key and 'True' as the value.
        Basically 'vis' stores the (visited_vertices, cost) from the current vertices as the (key, value) pair.
        The while loop executes until the queue 'q' is empty.
        The first vertex in the queue is removed and stored in the variable 'front'.
        Then for loop iterates over each edge in front's adjacency list.
        If the destination vertex of the current edge is not in the vertexDown set and is not already in the 'vis' dictionary, then its name is added to the edges list and it is added to the queue 'q'.
        The destination vertex is added to the 'vis' dictionary with the edge weight as its value.
        Then the edges are sorted in ascending order and are written to a output file.

        Overall, this code is using breadth-first search (BFS) to traverse the graph and find all vertices that are reachable from the current vertex.
        The time complexity of this algorithm is O(V^2 + VE), where V is the number of vertices and E is the number of edges in the graph.
        """
        vertices = sorted(self.vertexMap.keys())
        # O(V^2 + VE)
        for vertexName in vertices:
            v = self.getVertex(vertexName)
            if v not in self.vertexDown:
                self.file.write(vertexName+"\n")
                edges = []
                # O(V+E)
                q = [v]
                vis = {v: True}
                while q:
                    front = q.pop(0)
                    for edge in front.adj:
                        if edge[0] not in self.vertexDown and edge[0] not in vis:
                            edges.append(edge[0].name)
                            vis[edge[0]] = edge[1]
                            q.append(edge[0])
                edges.sort()
                for edge in edges:
                    self.file.write(f"  {edge}\n")
        self.file.write("\n")

    # Function to perform Print command. Print's the current state of the graph
    def printAll(self):
        vertices = sorted(self.vertexMap.keys())
        for vertexName in vertices:
            text = vertexName
            v = self.getVertex(vertexName)
            if v in self.vertexDown:
                text += " DOWN"
            self.file.write(text+"\n")
            edges = copy.deepcopy(v.adj)
            if v in self.edgeDown:
                edges.extend(self.edgeDown[v])
            edges.sort()
            for edge in edges:
                edText = f"  {edge[0].name}" + " {:.2f}".format(edge[1])
                if self.containsEdgeDown(v, edge[0]):
                    edText += " DOWN"
                self.file.write(edText+"\n")
        self.file.write("\n")

    # Function to perofrm the Path query
    def pathFinder(self, fromNode, toNode):
        dist = self.calculatePath(fromNode, toNode)
        self.printPath(dist)

    # Function to print the shortest path between given vetices
    def printPath(self, dist):
        names = [item.name for item in dist[1]]
        self.file.write(" ".join(names) + " {:.2f}".format(dist[0]) + "\n\n")

    # This function proces the standard input and then according to that call the respective query functions
    def processRequest(self):
        try:
            commands = {
                "print": self.printAll,
                "path": self.pathFinder,
                "edgedown": self.markEdgeDown,
                "vertexdown": self.markVertexDown,
                "edgeup": self.markEdgeUp,
                "vertexup": self.markVertexUp,
                "deleteedge": self.deleteEdge,
                "addedge": self.addEdge,
                "reachable": self.printReachable
            }

            # If the input is from a file
            if (len(sys.argv) >= 3):
                queryIn = sys.argv[2]
                with open(queryIn) as file:
                    for line in file:
                        # print(line)
                        inputOp = line.split()
                        if inputOp[0].lower() in ["print", "reachable"]:
                            commands[inputOp[0].lower()]()
                        elif inputOp[0].lower() == "quit":
                            return
                        elif inputOp[0].lower() in ["vertexup", "vertexdown"]:
                            commands[inputOp[0].lower()](inputOp[1])
                        elif inputOp[0].lower() == "addedge":
                            commands[inputOp[0].lower()](
                                inputOp[1], inputOp[2], float(inputOp[3]))
                        elif inputOp[0].lower() in ["path", "deleteedge", "edgedown", "edgeup"]:
                            commands[inputOp[0].lower()](inputOp[1], inputOp[2])
                        else:
                            print("Error. Wrong query passed. check again")


            # User input
            else:
                print("\n\n AFTER YOU ARE DONE WITH THE QUERIES, PASS THE 'QUIT' COMMAND TO EXIT FROM THE CODE. THE OUTPUT OF THIS PROGRAM WILL BE STORED IN A FILE 'OUTPUT.txt'.\n\n")
                while True:
                    inputOp = input("Enter the Command here: ").split()
                    if inputOp[0].lower() in ["print", "reachable"]:
                        commands[inputOp[0].lower()]()
                    elif inputOp[0].lower() == "quit":
                        return
                    elif inputOp[0].lower() in ["vertexup", "vertexdown"]:
                        commands[inputOp[0].lower()](inputOp[1])
                    elif inputOp[0].lower() == "addedge":
                        commands[inputOp[0].lower()](
                            inputOp[1], inputOp[2], float(inputOp[3]))
                    elif inputOp[0].lower() in ["path", "deleteedge", "edgedown", "edgeup"]:
                        commands[inputOp[0].lower()](inputOp[1], inputOp[2])
                    else:
                        print("Error. Wrong query passed. check again")

        except Exception as e:
            print(e)


    # Function to perform EdgeDown Query
    def markEdgeDown(self, fromNode, toNode):
        v = self.getVertex(fromNode)
        u = self.getVertex(toNode)
        for i in range(len(v.adj)):
            if u == v.adj[i][0]:
                if v in self.edgeDown:
                    self.edgeDown[v].append(v.adj[i])
                else:
                    self.edgeDown[v] = [v.adj[i]]
                del v.adj[i]
                break
    
    # Function to perform EdgeUp Query
    def markEdgeUp(self, fromNode, toNode):
        v = self.getVertex(fromNode)
        u = self.getVertex(toNode)
        adj = [edge[0] for edge in v.adj]
        if u not in adj:
            for i in range(len(self.edgeDown[v])):
                if self.edgeDown[v][i][0] == u:
                    v.adj.append(self.edgeDown[v][i])
                    del self.edgeDown[v][i]
                    break
    
    # Function to perform VertexDown Query
    def markVertexDown(self, name):
        v = self.getVertex(name)
        self.vertexDown[v] = name
    
    # Function to perform VertexUp Query
    def markVertexUp(self, name):
        v = self.getVertex(name)
        if v in self.vertexDown:
            del self.vertexDown[v]

    # Function to perform DeleteEdge query
    def deleteEdge(self, fromNode, toNode):
        v = self.getVertex(fromNode)
        u = self.getVertex(toNode)
        if v in self.edgeDown:
            for i in range(len(self.edgeDown[v])):
                if u == self.edgeDown[i][0]:
                    del self.edgeDown[i]
                    break
        else:
            for i in range(len(v.adj)):
                if u == v.adj[i][0]:
                    del v.adj[i]
                    break

# Main function
def main():
    g = Graph()
    g.setFile("output.txt")
    fin = sys.argv[1]
    with open(fin) as f:
        lines = f.readlines()

    #  Read the edges and insert
    for line in lines:
        line = line.strip().split(" ")
        if (len(line) != 3):
            print("Skipping ill-formatted line ", line)
            continue
        source = line[0]
        dest = line[1]
        cost = float(line[2])
        g.addEdge(source, dest, cost)
        g.addEdge(dest, source, cost)
    g.processRequest()
    g.closeFile()


if __name__ == "__main__":
    main()
