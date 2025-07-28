"""
Author: Ong Wei Han
Student ID: 33523401
"""

class Edge:
    def __init__(self, u, v, cost, time):
        """
        Function:
        Represents a directed edge in a graph with both cost and time weights.
        Stores the connection between two vertices (u -> v) with associated
        the associated weights which is the cost and time.

        :Input:
            u: Integer representing the source vertex ID
            v: Integer representing the destination vertex ID
            cost: Integer representing the traversal cost
            time: Integer representing the traversal time

        :Time Complexity: O(1)
        :Space Complexity: O(1)
        """
        self.u = u
        self.v = v
        self.cost = cost
        self.time = time


class Vertex:
    def __init__(self, id):
        """
        Function:
        Represents a vertex in a graph.

        Approach description:
        Initializes a vertex with default Dijkstra state (infinite cost/time,
        unvisited) and empty edge list. Tracks its position in the MinHeap.

        :Input:
            id: Integer identifier for the vertex

        :Time Complexity: O(1)
        :Space Complexity: O(1)
        """
        self.id = id
        self.edges = []
        self.index = None
        self.previous = None
        self.cost = float("inf")
        self.time = float("inf")
        self.visited = False

    def add_edge(self, edge):
        """
        Function:
        Adds a directed edge to this vertex's adjacency list.

        :Input:
            edge: Edge object to be added

        :Time Complexity: O(1)
        :Space Complexity: O(1)
        """
        self.edges.append(edge)


class MinHeap:
    def __init__(self, max_size):
        """
        Initializes a MinHeap object with a specified maximum size.
        The heap is represented as an array of Vertex objects, with the first element unused.

        Time Complexity:
            Worst case: O(1) - Initialization of array takes constant time.

        Space Complexity:
            O(L) - L being the number of vertices (location).
        """
        self.array = [None] * (max_size + 1)
        self.length = 0

    def __len__(self):
        """
        Returns the number of elements in the heap.

        Time Complexity:
            Worst case: O(1)

        Space Complexity:
            O(1) - No additional space is used.
        """
        return self.length

    def is_empty(self):
        """
        Returns True if the heap is empty, False otherwise.

        Time Complexity:
            Worst case: O(1) - Constant number of operations.

        Space Complexity:
            O(1) - No additional space is used.
        """
        return self.length == 0

    def rise(self, k):
        """
        Rise element at index k to its correct position
        :pre: 1 <= k <= self.length

        Time Complexity:
            Worst case: O(log L) - L being the length of the heap.
            The function may need to traverse the height of the heap.

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array
            Auxiliary space: O(1) - Constant space is used for item.

            Total Space Complexity: O(L) = O(L)
        """
        item = self.array[k]
        while k > 1 and is_lesser(item, self.array[k // 2]):
            self.array[k] = self.array[k // 2]
            self.array[k].index = k
            k = k // 2
        item.index = k
        self.array[k] = item

    def add(self, element):
        """
        Swaps elements while rising

        Time Complexity:
            Worst case: O(log L) - L being the length of the heap.
            rise method is called which is O(log L).

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array

            Total Space Complexity: O(L) = O(L)
        """
        if self.length + 1 == len(self.array):
            raise IndexError

        self.length += 1
        self.array[self.length] = element
        element.index = self.length
        self.rise(self.length)

    def smallest_child(self, k):
        """
        Returns the index of k's child with smallest value.
        :pre: 1 <= k <= self.length // 2

        Time Complexity:
            Worst case: O(1) - The function performs a constant number of operations.

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array

            Total Space Complexity: O(L) = O(L)
        """

        if 2 * k == self.length or is_lesser(self.array[2 * k], self.array[2 * k + 1]):
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k):
        """
        Make the element at index k sink to the correct position.
        :pre: 1 <= k <= self.length

        Time Complexity:
            Worst Case: O(logL) - L being the length of the heap.
            The function may need to traverse the height of the heap.

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array
            Auxiliary space: O(1) - Constant space is used for item.

            Total Space Complexity: O(L) = O(L)
        """
        item = self.array[k]

        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            if not is_lesser(self.array[min_child], item):
                break
            self.array[k] = self.array[min_child]
            self.array[k].index = k
            k = min_child

        item.index = k
        self.array[k] = item

    def serve(self):
        """
        Remove (and return) the minimum element from the heap.

        Time Complexity:
            Worst Case: O(log L) - L being the length of the heap.
            The sink method is called which is O(log L).

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array
            Auxiliary space: O(1) - Constant space is used for min_elem.

            Total Space Complexity: O(L) = O(L)
        """
        if self.is_empty():
            raise IndexError

        min_elem = self.array[1]
        self.length -= 1
        if self.length > 0:
            self.array[1] = self.array[self.length + 1]
            self.array[1].index = 1
            self.sink(1)

        min_elem.index = self.length + 1
        self.array[self.length + 1] = min_elem

        return min_elem

    def update(self, element):
        """
        Update the element in the heap and re-order it.

        Time Complexity:
            Worst Case: O(log L) - L being the length of the heap.
            The rise method is called which is O(log L).

        Space Complexity:
            Input space: O(L) - L being the number of vertices (location) from self.array in the rise method

            Total Space Complexity: O(L) = O(L)
        """
        if element.index is not None and element.index <= self.length:
            self.rise(element.index)


class Graph:
    def __init__(self, roads, no_of_vertices):
        """
        Function:
        Initializes a Graph object with vertices and edges constructed from the given roads.

        Approach description:
        This method constructs a graph by first initializing all vertices and then adding edges between them
        based on the input roads data. Each vertex is represented as a Vertex object, and edges are stored as
        Edge objects in the adjacency list of the corresponding vertices.

        :Input:
            roads: A list of tuples representing edges in the graph, where each tuple is of the form (u, v, cost, time).
                  Here, u and v are the id of the connected vertices while cost and time are the weight of the edge.
            no_of_vertices: An integer representing the total number of vertices in the graph.

        :Return:
            None

        :Time Complexity: O(L + R) where R is the number of edges (roads) and L is the number of vertices (location).

            Worst case analysis:
            - Vertex initialization: O(L) creation of L number of vertices.
            - Edge addition: O(R) each edge is processed once and appended to the adjacency list in O(1) time.

            Time Complexity: O(L) + O(R) = O(L + R).

        :Space Complexity: O(L + R) where R is the number of edges (roads) and L is the number of vertices (location).

            Auxiliary space analysis:
            - Vertices storage: O(L) storing L number of vertices.
            - Edges storage: O(R) each edge is stored once in the adjacency list of a vertex.

            Input space:
            - roads: O(R) R number of edges.

            Total: O(L) + O(R) + O(R) = O(L + R).
        """
        self.vertices = []
        for i in range(no_of_vertices):
            self.vertices.append(Vertex(i))

        for u, v, cost, time in roads:
            self.vertices[u].add_edge(Edge(u, v, cost, time))


def is_lesser(v1, v2):
    """
    Function:
    This function compares two vertices based on their cost and time attributes.

    Approach description:
    The function checks if the cost of vertex v1 is less than that of vertex v2.
    If the costs are equal, it then compares their time attributes.

    :Input:
        v1: A Vertex object representing the first vertex.
        v2: A Vertex object representing the second vertex.

    :return:
        True if v1 is less than v2 based on cost and time, False otherwise.

    :Time Complexity:
        O(1) - The function performs a constant number of operations regardless of input size.

    :Space Complexity:
        O(1) - The function uses a constant amount of space for its operations.
    """
    return v1.cost < v2.cost or (v1.cost == v2.cost and v1.time < v2.time)


def processing_stations(stations, friendStart):
    """
    Function:
    Processes stations list to calculate loop_time and adjust station times relative to a friend's starting point.

    Approach description:
    First, find the friend's starting station and resets its time to 0. Then, calculate the total loop_time
    as the sum of all station times. Finally, adjust other stations' times to be cumulative from the friend's start time.

    :Input:
        stations: A list of tuples (station_id, time) which represents stations and their associated times.
        friendStart: An integer which represents the vertex ID of the friend's starting station.

    :Return:
        Tuple (loop_time, friend_start_idx) where:
        - loop_time: An integer which represents the sum of all station times.
        - friend_start_idx: An integer which represents the index of the friend's starting station in the stations list.

    :Time Complexity: O(L) where L represents the number of vertices (locations)

        Worst case analysis:
        - Occurs when len(stations) is equals to the number of vertices in one graph of a layer.
        - Station processing: O(L) find friend's station and sum times.
        - Time adjustment: O(L) accumulate times all stations except the starting station.

        Time Complexity: O(L) + O(L) = O(L).

    :Space Complexity: O(L)
        Auxiliary space analysis:
        - Variable storage: O(1) constant extra space is used for variables (loop_time, friend_start_idx, temp, accumulated_time).
        - Modifies the input list in-place without additional storage.

        Input space:
        - stations: O(L) where L is the number of vertices (locations) in one graph of the layer_graph.

        Total Space Complexity = O(1) + O(L) = O(L).
    """

    loop_time = 0
    friend_start_idx = 0
    temp = 0
    accumulated_time = 0

    for i in range(len(stations)):
        id, time = stations[i]
        if id == friendStart:
            friend_start_idx = i
            temp = time
            stations[i] = (id, 0)
        loop_time += time

    for i in range(len(stations)):
        id, time = stations[i]
        if i != friend_start_idx:
            accumulated_time += temp
            temp = time
            stations[i] = (id, accumulated_time)

    return loop_time, friend_start_idx


def intercept(roads, stations, start, friendStart):
    """
    Function:
    Finds the most optimal interception point between a friend that is driving on roads
    and a friend that is taking a train on stations.

    Approach description:
    This method first processes the stations data to determine the total time (loop_time) to loop through
    all stations and back to the starting station.
    Additionally, the friends starting position in the stations list (friend_start_idx) is
    also determined within the processing.
    Then, a time-layered graph is constructed which represents a list of graphs where each
    layer corresponds to a unique time and the number of layers is bounded by loop_time.
    Then, every reachable vertex across all time layers is connected using Dijkstra's algorithm.
    Finally, it finds the optimal meeting point.

    :Input:
        roads: Represents a list of edges as tuples (u, v, cost, time) representing edges.
        stations: Represents a list of tuples (station_id, time) representing my friends' cyclic path of stations.
        start: Represents an integer ID of my starting location (vertex).
        friendStart: Represents an integer ID of my friends starting station (vertex).

    :Return:
        Tuple (cost, time, path) if a valid meeting exists, where:
        - cost: Total travel cost
        - time: Total travel time
        - path: Sequence of vertex IDs representing the shortest path to meet
        None if no valid meeting point exists

    :Time Complexity: O(R log L) where R is the number of edges (roads) and L is the number
                                of vertices (locations) in one graph of the layer_graph.

        Worst-case analysis:
        - Station processing: O(L) from processing_stations
        - Graph construction: O(L + R) (per layer, loop_time is constant)
        - Dijkstra's algorithm: O(R log L)
        - Path finding: O(V) (from shortest_path)

        Time Complexity: O(L) + O(L + R) + O(R log L) + O(V) = O(R log L)

    :Space Complexity: O(L + R)
        Auxiliary space analysis:
        - Station processing: O(1) (constant extra space for variables)
        - Graph storage: O(L + R) (per layer, loop_time is constant)
        - Dijkstra's algorithm: O(L) (MinHeap storage)

        Input space:
        - roads: O(R) (R number of edges)
        - stations: O(L) (up to L number of vertices in one graph of the layer_graph)

        Total Space Complexity: O(1) + O(L + R) + O(L) + O(R) + O(L) = O(L + R)
    """

    loop_time, friend_start_idx = processing_stations(stations, friendStart)

    # +1 bcos vertex start from 0 / no aux space cause max compares 2 numbers at a time whenever possible
    no_of_vertices = max(max(road[0], road[1]) for road in roads) + 1

    # creation of loop_time layers of graph
    layer_graph = [Graph(roads, no_of_vertices) for i in range(loop_time)]

    dijkstra(layer_graph, start, no_of_vertices, loop_time)

    return shortest_path(layer_graph, stations, loop_time, friend_start_idx)


def dijkstra(layer_graph, start, no_of_vertices, loop_time):
    """
    Function:
    This function runs the Dijkstra algorithm on the given layer_graph to find the shortest
    distance between the source every reachable vertices in each layer of the layer_graph.

    Approach description:
    This algorithm utilizes the multiverse concept where each verse is a layer of the graph.
    Each layer represents a different time in the graph which is bounded by the loop_time.
    So a layer graph acts like a 3D graph where the horizontal plane is the layer of a graph,
    and the vertical plane is the loop_time.
    MinHeap is used to find the shortest path between all the vertices in the layer graph.

    :Input:
        layer_graph: A list of Graph objects representing the time-dependent layers
        start: An integer that represents our starting vertex id
        no_of_vertices: An integer that represents the number of vertices in a graph
        loop_time: An integer (2 <= loop_time <= 100) that represents the time it takes to loop through all stations once.

    :Return:
        None

    :Time Complexity: O(R log L) where R is the number of edges (roads) and
                      L is the number of vertices (location) in one graph of the layer_graph.

        Worst case analysis:
        - Edge relaxation: O(R) - each edge is processed once.
        - Heap operations: O(log L) - each vertex is added/update to the heap once and removed once per edge.

        Time Complexity: O(R) * O(log L) = O(R log L)

    :Space Complexity: O(L + R) where L is the number of vertices (location)
                            and R is the number of edges (roads) in one graph of the layer_graph.

        Auxiliary space analysis:
        - MinHeap Storage: O(L) - storing L number of vertices in the heap across all layers but the number of
                                layers is a constant based on loop_time.

        Input space:
        - layer_graph: O(L + R) - storing L number of vertices and R number of edges in the adjacency list of
                                 a vertex for each layer but the number of layers is a constant based on loop_time.

        Total Space Complexity = O(L + R) + O(L) = O(L + R)
    """
    heap = MinHeap(no_of_vertices * loop_time)
    source = layer_graph[0].vertices[start]
    source.cost = 0
    source.time = 0
    heap.add(source)

    while not heap.is_empty():
        curr = heap.serve()
        if curr.visited:
            continue
        curr.visited = True

        for edge in curr.edges:
            new_cost = curr.cost + edge.cost
            new_time = curr.time + edge.time
            neighbour = layer_graph[new_time % loop_time].vertices[edge.v]

            if new_cost < neighbour.cost or (
                new_cost == neighbour.cost and new_time < neighbour.time
            ):
                neighbour.cost = new_cost
                neighbour.time = new_time
                neighbour.previous = curr
                if neighbour.index is None:
                    heap.add(neighbour)
                else:
                    heap.update(neighbour)


def shortest_path(layer_graph, stations, loop_time, friend_start_idx):
    """
    Function:
    Finds the optimal meeting point and path for two friends in a time-layered graph,
    where every reachable vertex is connected with the dijkstra method.

    Approach description:
    Iterates through all possible layers and stations to identify the earliest meeting point
    where both friends can arrive with minimal cost and time. The solution reconstructs the
    shortest path by backtracking from the meeting vertex.

    :Input:
        layer_graph: List of Graph objects representing time-dependent layers (loop_time bounds the layers).
        stations: List of tuples (station_id, arrival_time) defining the friends cyclic path.
        loop_time: An integer (2 <= loop_time <= 100) that represents the time it takes to loop through all stations once.
        friend_start_idx: Integer index for friend B's starting station in the stations list.

    :Return:
        - Tuple (cost, time, path) if a meeting point exists, where:
            - cost: Total travel cost from start.
            - time: Total travel time from start.
            - path: List of vertex IDs representing the shortest path to reach the optimal meeting point.
        - None if no valid meeting point exists.

    :Time Complexity: O(L) where N is the number of layers (bounded by loop_time, so it is a constant as L <= 100)
                    and L is the number of vertices (locations).

        Worst-case breakdown:
        - Layer iteration: O(N) -> O(1) loop_time is constant.
        - Station iteration: O(L) when len(stations) equals to the no of vertices in one graph of the layer_graph
        - Path reconstruction: O(L) backtracking to get the path and reversing the path list.

        Time Complexity: O(L) + O(L) = O(L)

    :Space Complexity: O(L + R) where L is the number of vertices (locations) and R is the number of edges (roads).

        Auxiliary space:
        - Path storage: O(L) storing L number of vertices.

        Input storage:
        - layer_graph: O(L + R) for the a graph in one layer of layer_graph but the number of layers is bounded by loop_time (constant).
        - stations: O(L) storing L number of vertices.

        Total Space Complexity = O(L + R) + O(L) + O(L) = O(L + R)

    """
    meeting_point = None
    for layer_index in range(loop_time):
        layer = layer_graph[layer_index]
        for station_idx in range(len(stations)):
            station = stations[(station_idx + friend_start_idx) % len(stations)]
            station_id = station[0]
            station_time = station[1] % loop_time
            vertex = layer.vertices[station_id]

            if (
                meeting_point is None
                or vertex.visited
                and (vertex.time % loop_time == station_time)
                and is_lesser(vertex, meeting_point)
            ):
                
                meeting_point = vertex

    if meeting_point.time != float("inf") and meeting_point.cost != float("inf"):
        path = []
        # backtracking from meeting_point to start
        curr = meeting_point
        while curr is not None:
            path.append(curr.id)
            curr = curr.previous
        path.reverse()
        return (meeting_point.cost, meeting_point.time, path)
    return None
