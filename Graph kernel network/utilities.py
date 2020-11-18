import sklearn.metrics as skm
import scipy.sparse as scis
import numpy as np

def generate_ball_neighbourhoods(vectors, r=1.0):
    """Connects input vectors into neighbourhoods based on Euclidean distance.

    Args:
        vectors (numpy array): A numpy array of Euclidean vectors.
        r (float, optional): Radius of the ball used to determine neighbourhoods of vectors. Defaults to 1.0.
    """
    distances = skm.pairwise_distances(vectors)
    distance_mask = distances <= r

    return distance_mask, distances



a = np.array([[0,0],[1,1],[3,3],[2.8,2.5]])
a = generate_ball_neighbourhoods(a)
print(scis.csr_matrix(a[0]))
