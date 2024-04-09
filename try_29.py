import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
random.seed(42)
# import pygraphviz as pgv

def auto_price(distance):
    return 10 * distance * (distance > 2) + 23

def local_price(distance):
    return distance

G1=nx.MultiDiGraph()
G2=nx.MultiDiGraph()

vertices = range(10)

# Define edges with information about whether it's local or auto

auto_edges = [(i, j, {'distance': np.random.randint(1,10), 'is_local': False}) for i in vertices for j in vertices if i != j]
local_train_routes = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]  # Updated local_train_routes
local_train_edges = [(u, v, {'distance': np.random.randint(1, 10), 'is_local': True}) for u, v in local_train_routes]
# auto_edges = [(i, j, {'distance': random.randint(1, 10), 'is_local': False}) for i in vertices for j in vertices if i < j and (i, j) not in [(3, 8)]]
# auto_edges.append((3, 8, {'distance': 100, 'is_local': False}))

# Add nodes and edges to the graph
G1.add_nodes_from(vertices)
G2.add_nodes_from(vertices)
G1.add_edges_from(auto_edges)
G2.add_edges_from(local_train_edges)

# Assign prices to edges
for u, v, d in G1.edges(data=True): 
    d['price'] = auto_price(d['distance'])

    print(f"Edge: ({u}, {v}), Attributes: {d}")    

for u,v,d in G2.edges(data=True):
        d['price'] = local_price(d['distance'])
        print(f"Edge: ({u}, {v}), Attributes: {d}")    
# Define optimization model (code omitted for brevity)

# Optimize the model (code omitted for brevity)

# Print solution (code omitted for brevity)

plt.show()
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):

    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
        

# Calculate positions of nodes using spring layout
pos = nx.spring_layout(G1)
pos1 = nx.spring_layout(G2)

# Visualizing the graph
plt.figure(figsize=(10, 6))

# Draw auto edges
auto_edges = [(u, v) for u, v, d in G1.edges(data=True)]
nx.draw_networkx_edges(G1, pos, edgelist=auto_edges, edge_color='black', width=1.0, alpha=0.5)


local_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G2.edges(data=True)}

# Draw local edges with different curvature
local_edges = [(u, v) for u, v, d in G2.edges(data=True)]
edge_weights = nx.get_edge_attributes(G2,'distance')

# curved_edge_labels = {edge: edge_weights[edge] for edge in local_edges}
curved_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G2.edges(data=True)}


for u, v in local_edges:
    curve_factor = 0.2  # Adjust the curve factor for local edges
    edge_pos = [(pos[u][0], pos[v][0]),
                ((1 - curve_factor) * pos[u][0] + curve_factor * pos[v][0],
                 (1 - curve_factor) * pos[u][1] + curve_factor * pos[v][1]),
                ((1 - curve_factor) * pos[v][0] + curve_factor * pos[u][0],
                 (1 - curve_factor) * pos[v][1] + curve_factor * pos[u][1]),
                (pos[v][0], pos[u][0])]
    nx.draw_networkx_edges(G2, pos, edgelist=[(u, v)], connectionstyle=f'arc3, rad = {curve_factor}',
                           edge_color='red', width=1.0, alpha=0.5,arrows=True)
    my_draw_networkx_edge_labels(G2, pos, edge_labels=curved_edge_labels,rotate=False,rad = curve_factor)



# Draw nodes with labels
nx.draw_networkx_nodes(G1, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_labels(G1, pos, font_size=10)

nx.draw_networkx_nodes(G2, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_labels(G2, pos, font_size=10)

# Add edge labels
edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G1.edges(data=True)}
nx.draw_networkx_edge_labels(G1, pos, edge_labels=edge_labels)


plt.title('Red is local black is auto')
plt.axis('off')

        
model = gp.Model("Path_Finding")

node_A = 3
node_B = 8

# Decision variables
x = {}
#y = {}
for i, j, data in G1.edges(data=True):
    x[i, j, False] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    x[j, i, False] =  model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    # y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
for i, j, data in G2.edges(data=True):
    x[i, j, True] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    x[j, i, True] =  model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    # y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")    

# Objective function: minimize total distance
# for i, j, data in G.edges(data=True):
#     print((i,j) , x[i,j], x[j,i] , data['distance'])
obj = (
    gp.quicksum(
        x[u, v, True] * d['price']for u, v, d in G2.edges(data=True))   
    + gp.quicksum(
        x[u, v, False] * d['price'] for u, v, d in G1.edges(data=True))
)
model.setObjective(obj, GRB.MINIMIZE)
# obj = gp.quicksum(x[u, v] * (G[u][v]['price']) for u, v, _ in G.edges(data=True) if u == node_A or v == node_B or (u, v) in local_train_routes or (v, u) in local_train_routes)
# model.setObjective(obj, GRB.MINIMIZE)


# model.setObjective(gp.quicksum(data['price'] * x[i, j] for i, j, data in G.edges(data=True)), GRB.MINIMIZE)

# for node in vertices:
#     model.addConstr(gp.quicksum(x[node, neighbor] for neighbor in G.neighbors(node) )  == 1  , f"Continuous_{node}")

# # Additional constraints based on x_ij and y_ij
# for i, j, data in G.edges(data=True):
#     model.addConstr(x[i, j] <= y[i, j], f"Edge_Local_{i}_{j}")

# Add constraint to ensure at most one edge between two nodes is selected and there is no looping
# for i, j in G.edges():
#     model.addConstr(x[i, j] +x[j, i] <= 1, f"AtMostOneEdge_{i}_{j}")

# 

# model.addConstr(gp.quicksum(x[node_A, neighbor] for neighbor in G.neighbors(node_A)) == 1, f"StartNode_{node_A}")
# model.addConstr(gp.quicksum(x[neighbor, node_B] for neighbor in G.neighbors(node_B)) == 1, f"EndNode_{node_B}")
# for node in G.nodes():
#     model.addConstr(gp.quicksum(x[i, j] for i, j in x.keys() if i == node) <= 1, f"Outgoing_edges_constraint_{node}")
#     model.addConstr(gp.quicksum(x[i, j] for i, j in x.keys() if j == node) <= 1, f"Incoming_edges_constraint_{node}")

# for node in G1.nodes() and G2.nodes():
#     if node != node_A and node != node_B:  # Exclude source and sink nodes
#         in_edges = [(u, v) for u, v in G1.in_edges(node)]
#         in_edges.append([(u, v) for u, v in G2.in_edges(node)])
#         out_edges = [(u, v) for u, v in G1.out_edges(node)]
#         out_edges.append([(u, v) for u, v in G2.out_edges(node)])
#         # Flow conservation constraint
#         model.addConstr(gp.quicksum(x[u, v] for u, v in in_edges) == gp.quicksum(x[u, v] for u, v in out_edges))
# for node in G1.nodes() and G2.nodes():
#     if node != node_A and node != node_B:  # Exclude source and sink nodes
#         in_edges = [(u, v) for u, v in G1.in_edges(node)]
#         in_edges.extend([(u, v) for u, v in G2.in_edges(node)])  # Use extend instead of append
#         out_edges = [(u, v) for u, v in G1.out_edges(node)]
#         out_edges.extend([(u, v) for u, v in G2.out_edges(node)])  # Use extend instead of append
#         # Check if in_edges and out_edges are empty

#         model.addConstr(gp.quicksum(x[u, v, ] for u, v in in_edges) == gp.quicksum(x[u, v] for u, v in out_edges))
# Add constraints for source node (node_A)
# Add constraints for source node (node_A)
# Add constraints for source node (node_A)
# Adding flow constraints for nodes in the combined graph
# Adding flow constraints for nodes in G1 and G2
for node in G1.nodes() | G2.nodes():  # Union of nodes from both graphs
    if node != node_A and node != node_B:  # Exclude source and sink nodes
        in_edges = [(u, v, is_local) for u, v, is_local in G1.in_edges(node, data="is_local")]
        in_edges.extend([(u, v, is_local) for u, v, is_local in G2.in_edges(node, data="is_local")])
        out_edges = [(u, v, is_local) for u, v, is_local in G1.out_edges(node, data="is_local")]
        out_edges.extend([(u, v, is_local) for u, v, is_local in G2.out_edges(node, data="is_local")])

        # Constraint: Flow conservation for each node
        model.addConstr(gp.quicksum(x[u, v, is_local] for u, v, is_local in in_edges) == gp.quicksum(x[u, v, is_local] for u, v, is_local in out_edges))



# Add constraints for sink node (node_B)
# for node in G1.nodes() and G2.nodes():
#     if node != node_A and node != node_B:  # Exclude source and sink nodes
#         in_edges = [(u, v, is_local) for u, v, is_local in G1.in_edges(node, data="is_local")]
#         in_edges.extend([(u, v, is_local) for u, v, is_local in G2.in_edges(node, data="is_local")])  # Use extend instead of append
#         out_edges = [(u, v, is_local) for u, v, is_local in G1.out_edges(node, data="is_local")]
#         out_edges.extend([(u, v, is_local) for u, v, is_local in G2.out_edges(node, data="is_local")])  # Use extend instead of append
        
        




# # Add constraints for source node (node_A)
# out_edges_source = [(u, v) for u, v in G1.out_edges(node_A)]
# out_edges_source.append([(u, v) for u, v in G2.out_edges(node_A)])
# model.addConstr(gp.quicksum(x[u, v] for u, v in out_edges_source) == 1)

# # Add constraints for sink node (node_B)
# in_edges_sink = [(u, v) for u, v in G1.in_edges(node_B)]
# in_edges_sink.append([(u, v) for u, v in G2.in_edges(node_B)])
# model.addConstr(gp.quicksum(x[u, v] for u, v in in_edges_sink) == 1)


# Optimize the model
model.optimize()

# Print solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for i, j, is_local in x:
        if x[i, j, is_local].x > 0.5:
            print(f"Edge from node {i} to node {j} is selected.")

plt.show()