import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from ts2vg import NaturalVG

# create a graph (you would load your own graph instead)
G = nx.karate_club_graph()
df = pd.read_csv("data\\DJI\\2000\\allyear2000.csv")
df_h = df.iloc[:, 1]
g = NaturalVG(directed=None).build(df_h)
G = g.as_networkx()

# get the k-hop neighbors of node_id
node_id = 100
k = 4
distances = nx.single_source_shortest_path_length(G, node_id)
k_hop_neighbors = [node for node, distance in distances.items() if distance == k]

# print the k-hop neighbors
print(k_hop_neighbors)
