import networkx as nx             # For creating and handling the graph
import matplotlib.pyplot as plt   # For drawing and saving visualisations
import heapq                      # For priority queue (used in A* open set)
import os                         # For creating output folder

# Create a folder to store step-by-step images
os.makedirs("a_star_steps", exist_ok=True)

# Create a directed graph (edges go only one way)
G = nx.DiGraph()

# Define the edges: (from_node, to_node, cost)
edges = [
    (1, 2, 7.81),
    (1, 3, 5.39),
    (3, 2, 4),
    (2, 8, 4),
    (3, 4, 5),
    (3, 5, 2.83),
    (3, 6, 4),
    (4, 6, 3),
    (5, 8, 2.83),
    (6, 7, 2.83),
    (8, 7, 2.83),
    (7, 9, 3.16),
    (8, 9, 3.16)
]

# Add each edge to the graph
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Set positions for drawing each node on the graph
# These control where each node appears on the plot
pos = {
    1: (2, 8),
    2: (7, 2),
    3: (7, 6),
    4: (11, 3),
    5: (9, 4),
    6: (11, 6),
    7: (13, 4),
    8: (11, 2),
    9: (14, 1)
}

# Define the heuristic (h(n)) for A*
# This is the estimated distance from each node to the goal (node 9)
heuristics = {
    1: 13.89,
    2: 7.07,
    3: 8.60,
    4: 3.61,
    5: 5.83,
    6: 5.83,
    7: 3.16,
    8: 3.16,
    9: 0
}

# A* search algorithm
def a_star(G, start, goal):

    # Priority queue for nodes to explore
    open_set = []

    # Push starting node with its f = h(n)
    heapq.heappush(open_set, (heuristics[start], start))

    # Dictionary to reconstruct the path later
    came_from = {}

    # Cost from start to each node
    g_score = {node: float('inf') for node in G.nodes}

    # Cost to start node is 0
    g_score[start] = 0

    # For tracking the number of steps / saving images
    step = 0

    # Main A* loop
    while open_set:

        # Get node with lowest f(n)
        f_current, current = heapq.heappop(open_set)

        # Draw the graph at this step
        draw_step(G, pos, g_score, open_set, current, came_from, step)

        # Increment step couter
        step += 1

        # If we reached the goal, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        # Explore all neighbors
        for neighbor in G.successors(current):

            # Calculate new cost to neighbor through current
            tentative_g = g_score[current] + G[current][neighbor]['weight']
            
            # If this path to neighbor is better, update it
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristics[neighbor]
                heapq.heappush(open_set, (f_score, neighbor))

    # If goal was never reached
    return None, float('inf')

# Visualisation function to draw each step
def draw_step(G, pos, g_score, open_set, current, came_from, step):
    
    plt.figure(figsize=(9, 6))

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=1000, font_weight='bold', arrows=True)

    # Highlight current node being expanded
    nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color='orange')

    # Highlight open set (frontier nodes)
    open_nodes = [node for _, node in open_set]
    nx.draw_networkx_nodes(G, pos, nodelist=open_nodes, node_color='skyblue')

    # Reconstruct path so far and highlight it
    path_nodes = []
    n = current
    while n in came_from:
        path_nodes.append(n)
        n = came_from[n]
    path_nodes.append(n)
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='red')

    # Draw weights (g values) on edges
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Save image for this step
    plt.title(f"A* Step {step}: Expanding Node {current}")
    plt.savefig(f"a_star_steps/step_{step}.png")
    plt.close()


# Run the A* search algorithm
path, total_cost = a_star(G, 1, 9)

# Output final result
print("Path found:", path)
print("Total distance:", round(total_cost, 2))