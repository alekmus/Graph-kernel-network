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


def build_connections(mesh, r):
    """Generates adjacency and edge feature matrices for nodes in a mesh
        based Euclidean neighbourhoods defined by a ball B(x,r) for each
        node x. 
    Args:
        mesh (numpy.ndarray): An array of nodes forming a mesh. 
        r (float): Radius of the desired ball neighbourhood.
    """
    distance_mask, distances = generate_ball_neighbourhoods(mesh, r)
    sparse_adjacency = scipy.sparse.csr_matrix(distance_mask)

    distances = distances[distance_mask]
    return sparse_adjacency, distances

a = np.array([[0,0],[1,1],[3,3],[2.8,2.5]])
print(build_connections(a,1))
