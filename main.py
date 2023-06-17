from typing import List
import random


# Adjacency List representation (graph)
class Node:
    def __init__(self, name: str):
        self.name = name
        self.nbrs = []  # List of adjacent neighbors to that city


class Graph:
    def __init__(self, cities: List[str]):
        self.m = {}
        for city in cities:
            self.m[city] = Node(city)

    def add_edge(self, x: str, y: str, undir=False):
        self.m[x].nbrs.append(y)
        # for undirected graph, both ways <--->
        if undir:
            self.m[y].nbrs.append(x)

    def print_adj_list(self):
        for city, node in self.m.items():
            print(city, "-->", end="")
            for nbr in node.nbrs:
                print(nbr, end=", ")
            print()


# Minimum Weaponry
def weaponry(prices_weapons: List[int], n: int) -> int:
    dp = [[0] * (n+1) for _ in range(n+1)]

    for i in range(n-1, -1, -1):
        for j in range(n):
            if i == j:
                dp[i][i] = n * prices_weapons[i]
            elif i < j:
                y = n - (j - i)
                pick_left = prices_weapons[i] * y + dp[i+1][j]
                pick_right = prices_weapons[j] * y + dp[i][j-1]

                dp[i][j] = min(pick_left, pick_right)

    for row in dp:
        print(" ".join(map(str, row)))

    print("Minimum Money To Be Spend (in 10 crores): ")
    return dp[0][n-1]


# Path Escape
def solve_maze(maze: List[List[int]]) -> bool:
    N = len(maze)  # Update N based on the size of the maze
    sol = [[0] * N for _ in range(N)]

    if not solve_maze_util(maze, 0, 0, sol, N):  # Pass N as an argument
        print("Path doesn't exist. Hold on, Military AirBorne")
        return False

    print("Escape as follows:\n")
    for row in sol:
        print(" ".join(map(str, row)))

    return True

# Path Escape
def solve_maze_util(maze: List[List[int]], t: int, s: int, sol: List[List[int]], N: int) -> bool:
    if t == N - 1 and s == N - 1 and maze[t][s] == 1:
        sol[t][s] = 1
        return True

    if 0 <= t < N and 0 <= s < N and maze[t][s] == 1:
        if sol[t][s] == 1:
            return False

        sol[t][s] = 1

        if solve_maze_util(maze, t + 1, s, sol, N):
            return True

        if solve_maze_util(maze, t, s + 1, sol, N):
            return True

        sol[t][s] = 0
        return False

    return False



# Floyd Warshall Algorithm
# Floyd Graph
def floyd_graph():
    V = 4  # Update the number of vertices based on the graph size
    INF = float('inf')  # Update the definition of INF

    graph = [[0, 1, INF, INF],
             [INF, 0, 1, INF],
             [INF, INF, 0, 1],
             [INF, INF, INF, 0]]

    floyd_warshall(graph, V)


# Floyd Warshall Algorithm
def print_floyd(dist: List[List[int]], V: int):
    print("Shortest distances between every pair of vertices")
    for i in range(V):
        for j in range(V):
            if dist[i][j] == INF:
                print("INF", end="\t")
            else:
                print(dist[i][j], end="\t")
        print()


def floyd_warshall(graph: List[List[int]], V: int):
    dist = [[0] * V for _ in range(V)]

    for i in range(V):
        for j in range(V):
            dist[i][j] = graph[i][j]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    print_floyd(dist, V)







# Comparator function
def cmp(val):
    return val

# Sort Map
def sort(mp):
    sorted_map = sorted(mp.items(), key=lambda x: cmp(x[1]))
    return sorted_map



# Binary Search
def binary_search(arr: List[str], target: str) -> int:
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# Constants
N = 16
V = 16
INF = 99999

# Main Function
if __name__ == "__main__":
    # Graph example
    cities = ['A', 'B', 'C', 'D']
    g = Graph(cities)
    g.add_edge('A', 'B')
    g.add_edge('B', 'C')
    g.add_edge('C', 'D')
    g.print_adj_list()

    # Minimum Weaponry example
    weapon_prices = [10, 20, 30, 40]
    print("Minimum Money To Be Spend (in 10 crores):", weaponry(weapon_prices, len(weapon_prices)))

    # Path Escape example
    maze = [[1, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1]]
    solve_maze(maze)

    # Floyd Warshall example
    floyd_graph()

    # Sorting Map example
    weapon_count = {'A': 10, 'B': 20, 'C': 5, 'D': 15}
    sorted_weapon_count = sort(weapon_count)
    print("Sorted Weapon Count:")
    for item in sorted_weapon_count:
        print(item[0], ":", item[1])

    # Binary Search example
    cities = ['A', 'B', 'C', 'D', 'E', 'F']
    target_city = 'D'
    result = binary_search(cities, target_city)
    if result != -1:
        print("City", target_city, "found at index", result)
    else:
        print("City", target_city, "not found")
