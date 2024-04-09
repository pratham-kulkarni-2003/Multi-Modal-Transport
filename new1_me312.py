import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def auto_price(distance):
    return 10 * distance * (distance > 2) + 23

def local_price(distance):
    return distance

G = nx.Graph()

vertices = range(10)

# Define edges with information about whether it's local or auto
auto_edges = [(i, j, {'distance': np.random.randint(1,10), 'is_local': False}) for i in vertices for j in vertices if i != j]
local_train_routes = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]  # Updated local_train_routes
local_train_edges = [(u, v, {'distance': np.random.randint(1, 10), 'is_local': True}) for u, v in local_train_routes]

# Add nodes and edges to the graph
G.add_nodes_from(vertices)
G.add_edges_from(auto_edges + local_train_edges)

# Assign prices and distances to edges
for u, v, d in G.edges(data=True):
    if d['is_local']:
        d['distance'] = np.random.randint(1, 10)
        d['price'] = local_price(d['distance'])
    else:
        d['distance'] = 10
        d['price'] = auto_price(d['distance'])

# def find_all_paths(graph, start, end, path=[], is_local=False):
#     path = path + [(start, is_local)]  # Include the is_local flag in the path tuple
#     if start == end:
#         return [path]
#     if not graph.has_node(start):
#         return []
#     paths = []
#     for node in graph.neighbors(start):
#         if node not in [p[0] for p in path]:
#             # Check if the edge is part of local_train_routes
#             is_local_edge = (start, node) in local_train_routes or (node, start) in local_train_routes
#             new_paths = find_all_paths(graph, node, end, path, is_local=is_local_edge)
#             for new_path in new_paths:
#                 paths.append(new_path)
#     return paths

# def optimize_path(graph, local_train_routes, input_station, output_station):
#     model = gp.Model("path_optimization")

#     # Decision variables
#     x = {}
#     for u, v, _ in graph.edges(data=True):
#         x[u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

#     # Add no self-looping constraint
#     for u in graph.nodes():
#       model.addConstr(x.get((u, u), 0) == 0, name=f"no_self_loop_constraint_{u}")

#     # Objective function
#     obj = gp.quicksum(x[u, v] * (d['price'] if d['is_local'] else auto_price(d['distance'])) for u, v, d in graph.edges(data=True))
#     model.setObjective(obj, GRB.MINIMIZE)

#     # Optimize model
#     model.optimize()
#     return \

def optimize_path(graph, local_train_routes, input_station, output_station):
    model = gp.Model("path_optimization")

    # Decision variables
    x = {}
    for u, v, _ in graph.edges(data=True):
        x[u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

    # Add constraints
    model.addConstr(gp.quicksum(x.get((input_station, v), 0) for v in graph.neighbors(input_station)) == 1, name="start_node_constraint")
    model.addConstr(gp.quicksum(x.get((u, output_station), 0) for u in graph.neighbors(output_station)) == 1, name="end_node_constraint")

    # No self-looping constraint
    for u in graph.nodes():
        model.addConstr(x.get((u, u), 0) == 0, name=f"no_self_loop_constraint_{u}")

    # Objective function
    obj = gp.quicksum(x[u, v] * (graph[u][v]['price'] if graph[u][v]['is_local'] else auto_price(graph[u][v]['distance'])) for u, v, _ in graph.edges(data=True) if u == input_station or v == output_station or (u, v) in local_train_routes or (v, u) in local_train_routes)
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()
    # model.setObjective(gp.quicksum(data['distance'] * x[i, j] for i, j, data in G.edges(data=True)), GRB.MINIMIZE)
    # Retrieve optimized solution
    optimized_path = [(u, v) for (u, v) in x if x[u, v].X > 0.5]
    optimized_cost = model.objVal
    optimized_distance = sum(graph[u][v]['distance'] for u, v in optimized_path)

    return optimized_path, optimized_cost, optimized_distance

# input_station = int(input("Enter the input station (0-9): "))
# output_station = int(input("Enter the output station (0-9): "))


def optimize_path1(graph, local_train_routes, input_station, output_station):
    model = gp.Model("path_optimization")

    # Decision variables
    x = {}
    for u, v, _ in graph.edges(data=True):
        x[u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

    # Add constraints
    model.addConstr(gp.quicksum(x.get((input_station, v), 0) for v in graph.neighbors(input_station)) == 1, name="start_node_constraint")
    model.addConstr(gp.quicksum(x.get((u, output_station), 0) for u in graph.neighbors(output_station)) == 1, name="end_node_constraint")

    # No self-looping constraint
    for u in graph.nodes():
        model.addConstr(x.get((u, u), 0) == 0, name=f"no_self_loop_constraint_{u}")

    # Objective function
    obj = gp.quicksum(x[u, v] * (graph[u][v]['price'] if graph[u][v]['is_local'] else auto_price(graph[u][v]['distance'])) for u, v in x)
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    # Retrieve optimized solution
    optimized_path = []
    for u, v in graph.edges():
        if x[u, v].X > 0.5:
            optimized_path.append((u, v))

    optimized_cost = model.objVal
    optimized_distance = sum(graph[u][v]['distance'] for u, v in optimized_path)

    return optimized_path, optimized_cost, optimized_distance

# Assuming you have defined G and local_train_routes appropriately
input_station = int(input("Enter the input station (0-9): "))
output_station = int(input("Enter the output station (0-9): "))

optimized_path, optimized_cost, optimized_distance = optimize_path1(G, local_train_routes, input_station, output_station)
print("Optimized Path:", optimized_path)
print("Optimized Cost:", optimized_cost)
print("Optimized Distance:", optimized_distance)