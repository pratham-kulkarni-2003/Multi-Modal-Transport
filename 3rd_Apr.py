import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
# random.seed(9001)

def auto_price(distance):
    return 10 * distance * (distance > 2) + 23

def local_price(distance):
    return distance

G=nx.MultiDiGraph()

vertices = range(15)

# Define edges with information about whether it's local or auto

auto_edges = [(i, j, {'distance': np.random.randint(10,30), 'is_local': False}) for i in vertices for j in vertices if i != j]
local_train_routes = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9), (9,10), (10,11), (11,12), (12,13), (13,14)]  # Updated local_train_routes
local_train_edges = [(u, v, {'distance': np.random.randint(10, 30), 'is_local': True}) for u, v in local_train_routes]
# auto_edges = [(i, j, {'distance': random.randint(1, 10), 'is_local': False}) for i in vertices for j in vertices if i < j and (i, j) not in [(3, 8)]]
# auto_edges.append((3, 8, {'distance': 100, 'is_local': False}))

# Add nodes and edges to the graph
G.add_nodes_from(vertices)

G.add_edges_from(auto_edges)
G.add_edges_from(local_train_edges)

# Assign prices to edges
for u, v, d in G.edges(data=True): 
    d['price'] = auto_price(d['distance'])

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
pos = nx.spring_layout(G)
pos1 = nx.spring_layout(G)

# Visualizing the graph
plt.figure(figsize=(10, 6))

# Draw auto edges
# auto_edges = [(u, v) for u, v, d in G.edges(data=True)]
# nx.draw_networkx_edges(G, pos, edgelist=auto_edges, edge_color='black', width=1.0, alpha=0.5)


# local_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True)}

# # Draw local edges with different curvature
# local_edges = [(u, v) for u, v, d in G.edges(data=True)]
# edge_weights = nx.get_edge_attributes(G,'distance')

# # curved_edge_labels = {edge: edge_weights[edge] for edge in local_edges}
# curved_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True)}


# for u, v in local_edges:
#     curve_factor = 0.2  # Adjust the curve factor for local edges
#     edge_pos = [(pos[u][0], pos[v][0]),
#                 ((1 - curve_factor) * pos[u][0] + curve_factor * pos[v][0],
#                  (1 - curve_factor) * pos[u][1] + curve_factor * pos[v][1]),
#                 ((1 - curve_factor) * pos[v][0] + curve_factor * pos[u][0],
#                  (1 - curve_factor) * pos[v][1] + curve_factor * pos[u][1]),
#                 (pos[v][0], pos[u][0])]
#     nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3, rad = {curve_factor}',
#                            edge_color='red', width=1.0, alpha=0.5,arrows=True)
#     my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rotate=False,rad = curve_factor)



# # Draw nodes with labels
# nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
# nx.draw_networkx_labels(G, pos, font_size=10)

# nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
# nx.draw_networkx_labels(G, pos, font_size=10)

# # Add edge labels
# edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


# plt.title('Red is local black is auto')
# plt.axis('off')

        
model = gp.Model("Path_Finding")

node_A = 3
node_B = 13
# def auto_price(distance):
#     return 10 * distance * (distance > 2) + 23

# def local_price(distance):
#     return distance

# G=nx.MultiDiGraph()

# vertices = range(5)

# # Define edges with information about whether it's local or auto
# auto_edges = [(i, j, {'distance': 10, 'is_local': False}) for i in vertices for j in vertices if i != j]
# local_train_routes = [(0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]  # Updated local_train_routes
# local_train_edges = [(u, v, {'distance': np.random.randint(1, 10), 'is_local': True}) for u, v in local_train_routes]

# # auto_edges = [(i, j, {'distance': random.randint(1,30), 'is_local': False}) for i in vertices for j in vertices if i < j ]

# local_train_routes = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Updated local_train_routes
# local_train_edges = [(u, v, {'distance': random.randint(1,30), 'is_local': True}) for u, v in local_train_routes]

# # Add nodes and edges to the graph
# G.add_nodes_from(vertices)
# G.add_edges_from(auto_edges)
# G.add_edges_from(local_train_edges)
# nx.degree(G)
# # Assign prices and distances to edges
# for u, v, d in G.edges(data=True):
#     # print(u,v,d)
#     #d represents the dictionary, each edge is stored as a tuple of int, int, dictionary
#     if d['is_local']:
#         d['price'] = 10 #local_price(d['distance'])
#         #update the local price because we dk the model
#     else:
#         d['price'] = auto_price(d['distance'])
     

        
# def my_draw_networkx_edge_labels(
#     G,
#     pos,
#     edge_labels=None,
#     label_pos=0.5,
#     font_size=10,
#     font_color="k",
#     font_family="sans-serif",
#     font_weight="normal",
#     alpha=None,
#     bbox=None,
#     horizontalalignment="center",
#     verticalalignment="center",
#     ax=None,
#     rotate=True,
#     clip_on=True,
#     rad=0
# ):
#     """Draw edge labels.

#     Parameters
#     ----------
#     G : graph
#         A networkx graph

#     pos : dictionary
#         A dictionary with nodes as keys and positions as values.
#         Positions should be sequences of length 2.

#     edge_labels : dictionary (default={})
#         Edge labels in a dictionary of labels keyed by edge two-tuple.
#         Only labels for the keys in the dictionary are drawn.

#     label_pos : float (default=0.5)
#         Position of edge label along edge (0=head, 0.5=center, 1=tail)

#     font_size : int (default=10)
#         Font size for text labels

#     font_color : string (default='k' black)
#         Font color string

#     font_weight : string (default='normal')
#         Font weight

#     font_family : string (default='sans-serif')
#         Font family

#     alpha : float or None (default=None)
#         The text transparency

#     bbox : Matplotlib bbox, optional
#         Specify text box properties (e.g. shape, color etc.) for edge labels.
#         Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

#     horizontalalignment : string (default='center')
#         Horizontal alignment {'center', 'right', 'left'}

#     verticalalignment : string (default='center')
#         Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

#     ax : Matplotlib Axes object, optional
#         Draw the graph in the specified Matplotlib axes.

#     rotate : bool (deafult=True)
#         Rotate edge labels to lie parallel to edges

#     clip_on : bool (default=True)
#         Turn on clipping of edge labels at axis boundaries

#     Returns
#     -------
#     dict
#         dict of labels keyed by edge

#     Examples
#     --------
#     >>> G = nx.dodecahedral_graph()
#     >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

#     Also see the NetworkX drawing examples at
#     https://networkx.org/documentation/latest/auto_examples/index.html

#     See Also
#     --------
#     draw
#     draw_networkx
#     draw_networkx_nodes
#     draw_networkx_edges
#     draw_networkx_labels
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     if ax is None:
#         ax = plt.gca()
#     if edge_labels is None:
#         labels = {(u, v): d for u, v, d in G.edges(data=True)}
#     else:
#         labels = edge_labels
#     text_items = {}
#     for (n1, n2), label in labels.items():
#         (x1, y1) = pos[n1]
#         (x2, y2) = pos[n2]
#         (x, y) = (
#             x1 * label_pos + x2 * (1.0 - label_pos),
#             y1 * label_pos + y2 * (1.0 - label_pos),
#         )
#         pos_1 = ax.transData.transform(np.array(pos[n1]))
#         pos_2 = ax.transData.transform(np.array(pos[n2]))
#         linear_mid = 0.5*pos_1 + 0.5*pos_2
#         d_pos = pos_2 - pos_1
#         rotation_matrix = np.array([(0,1), (-1,0)])
#         ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
#         ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
#         ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
#         bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
#         (x, y) = ax.transData.inverted().transform(bezier_mid)

#         if rotate:
#             # in degrees
#             angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
#             # make label orientation "right-side-up"
#             if angle > 90:
#                 angle -= 180
#             if angle < -90:
#                 angle += 180
#             # transform data coordinate angle to screen coordinate angle
#             xy = np.array((x, y))
#             trans_angle = ax.transData.transform_angles(
#                 np.array((angle,)), xy.reshape((1, 2))
#             )[0]
#         else:
#             trans_angle = 0.0
#         # use default box of white with white border
#         if bbox is None:
#             bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
#         if not isinstance(label, str):
#             label = str(label)  # this makes "1" and 1 labeled the same

#         t = ax.text(
#             x,
#             y,
#             label,
#             size=font_size,
#             color=font_color,
#             family=font_family,
#             weight=font_weight,
#             alpha=alpha,
#             horizontalalignment=horizontalalignment,
#             verticalalignment=verticalalignment,
#             rotation=trans_angle,
#             transform=ax.transData,
#             bbox=bbox,
#             zorder=1,
#             clip_on=clip_on,
#         )
#         text_items[(n1, n2)] = t

#     ax.tick_params(
#         axis="both",
#         which="both",
#         bottom=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False,
#     )

#     return text_items
        

# # Calculate positions of nodes using spring layout
# pos = nx.spring_layout(G)

# # Visualizing the graph
# plt.figure(figsize=(10, 6))

# # Draw auto edges
auto_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['is_local']]
nx.draw_networkx_edges(G, pos, edgelist=auto_edges, edge_color='black', width=1.0, alpha=0.5)


local_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if d['is_local']}

# Draw local edges with different curvature
local_edges = [(u, v) for u, v, d in G.edges(data=True) if d['is_local']]
edge_weights = nx.get_edge_attributes(G,'distance')

# curved_edge_labels = {edge: edge_weights[edge] for edge in local_edges}
curved_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if d['is_local']}


for u, v in local_edges:
    curve_factor = 0.2  # Adjust the curve factor for local edges
    edge_pos = [(pos[u][0], pos[v][0]),
                ((1 - curve_factor) * pos[u][0] + curve_factor * pos[v][0],
                 (1 - curve_factor) * pos[u][1] + curve_factor * pos[v][1]),
                ((1 - curve_factor) * pos[v][0] + curve_factor * pos[u][0],
                 (1 - curve_factor) * pos[v][1] + curve_factor * pos[u][1]),
                (pos[v][0], pos[u][0])]
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3, rad = {curve_factor}',
                           edge_color='red', width=1.0, alpha=0.5,arrows=True)
    my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rotate=False,rad = curve_factor)



# Draw nodes with labels
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_labels(G, pos, font_size=10)

# Add edge labels
edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if not d['is_local']}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


plt.title('Red is local black is auto')
plt.axis('off')

        
model = gp.Model("Path_Finding")



# Decision variables
x = {}
# y = {}
for i, j, data in G.edges(data=True):
    x[i, j , str(data["is_local"])] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    x[j, i , str(data["is_local"])] =  model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    # y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

# # Objective function: minimize total distance
# print("here")
# print(len(x))  #gives 28
# # print(x)
# print(len(G.edges(data=True)))   #gives 14 as the number of edges in the graph
# # for i, j, data in G.edges(data=True):
# #     print((i,j) , data['distance'] , data["is_local"] , data["price"])



model.setObjective(gp.quicksum(data['price'] * x[i, j , str(data["is_local"])] for i, j, data in G.edges(data=True)) + gp.quicksum(data['price'] * x[i, j, str(data["is_local"])] for j, i, data in G.edges(data=True)) , GRB.MINIMIZE)




list_of_xs_for_c1 = []

for neighbor in vertices:
    if neighbor!=node_A:
        for i, j, data in G.edges(data=True):
            if i==node_A and j==neighbor:
                list_of_xs_for_c1.append([node_A , neighbor , str(data["is_local"])])
            if j==node_A and i==neighbor:
                list_of_xs_for_c1.append([node_A , neighbor , str(data["is_local"])])

# print(list_of_xs_for_c1)

model.addConstr( gp.quicksum(x[i[0], i[1] , i[2]] for i in list_of_xs_for_c1)   == 1 , f"StartNode{node_A}")


list_of_xs_for_c2 = []

for neighbor in vertices:
    if neighbor!=node_B:
        for i, j, data in G.edges(data=True):
            if i==neighbor and j==node_B:
                list_of_xs_for_c2.append([neighbor , node_B , str(data["is_local"])])
            if j==neighbor and i==node_B:
                list_of_xs_for_c2.append([neighbor , node_B , str(data["is_local"])])

# print(list_of_xs_for_c2)

model.addConstr( gp.quicksum(x[i[0], i[1] , i[2]] for i in list_of_xs_for_c2)   == 1 , f"EndNode_{node_B}")  


print()
print()

list_of_xs_for_c3 = []

for (i_mid, j_mid,local_mid) in x:
    for (i_before, j_before,local_before) in x:
        for (i_after, j_after,local_after) in x:
            if j_before==i_mid and i_after==j_mid and i_before!=j_mid and j_after!=i_mid:
                list_of_xs_for_c3.append([ [i_before, j_before,local_before] ,[i_mid, j_mid,local_mid] , [i_after, j_after,local_after]])

                # print([ [i_before, j_before,local_before] ,[i_mid, j_mid,local_mid] , [i_after, j_after,local_after]])

for i in list_of_xs_for_c3: 
    if i[1][0]!=node_A and i[1][1]!=node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[0][0], j[0][1] , j[0][2]]+x[j[2][0], j[2][1] , j[2][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 2 ) , f"c3")  
    if i[1][0]==node_A and i[1][1]!=node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[2][0], j[2][1] , j[2][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 1) , f"c4")  
    if i[1][0]!=node_A and i[1][1]==node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[0][0], j[0][1] , j[0][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 1) , f"c5")  


# Optimize the model
model.optimize()
# Print solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for i, j , is_local in x:
        if x[i, j , is_local].x > 0.5:
            print(f"Edge from node {i} to node {j} via ",is_local," is selected.")
else:
    print("No solution found.")

plt.show()