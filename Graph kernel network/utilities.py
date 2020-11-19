import sklearn.metrics as skm
import scipy.sparse
import numpy as np

def generate_ball_neighbourhoods(vectors, r):
    """Connects input vectors into neighbourhoods based on Euclidean distance.

    Args:
        vectors (numpy.ndarray): An array of Euclidean vectors.
        r (float): Radius of the ball used to determine neighbourhoods of vectors. 
    """
    distances = skm.pairwise_distances(vectors)
    distance_mask = distances <= r

    return distance_mask, distances


def build_connections(mesh, r, self_loops_allowed=False):
    """Generates adjacency and edge feature matrices for nodes in a mesh
        based Euclidean neighbourhoods defined by a ball B(x,r) for each
        node x. 
    Args:
        mesh (numpy.ndarray): An array of nodes forming a mesh. 
        r (float): Radius of the desired ball neighbourhood.
        self_loops_allowed (boolean, optional): Boolean determining if nodes
                                                should have connection to themselves. 
                                                Defaults to False.
    """
    distance_mask, distances = generate_ball_neighbourhoods(mesh, r)

    if(not self_loops_allowed):
        np.fill_diagonal(distance_mask, False)

    sparse_adjacency = scipy.sparse.csr_matrix(distance_mask)
    sparse_distances = scipy.sparse.csr_matrix(distances[distance_mask])
    return sparse_adjacency, sparse_distances


def generate_radial_mesh(n_nodes):
    nodes = [(0,0)]

    for i in range(n_nodes):
        nodes.append(0,i)

    return np.array(nodes)


import matplotlib.pyplot as plt


