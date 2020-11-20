from typing import Tuple
import sklearn.metrics as skm
import scipy.sparse
import numpy as np

def generate_ball_neighbourhoods(vectors, r) -> Tuple(np.ndarray, np.ndarray):
    """Connects input vectors into neighbourhoods based on Euclidean distance.

    Args:
        vectors (numpy.ndarray): An array of Euclidean vectors.
        r (float): Radius of the ball used to determine neighbourhoods of vectors. 

    Returns:
        Tuple: A dense boolean matrix that determines if the distance between two nodes
            is under r and a dense boolean matrix with actual distances.
    """
    distances = skm.pairwise_distances(vectors)
    distance_mask = distances <= r

    return distance_mask, distances


def build_connections(mesh, r, self_loops_allowed=False) -> Tuple(scipy.sparse.csr_matrix, scipy.sparse.csr_matrix):
    """Generates adjacency and edge feature matrices for nodes in a mesh
        based Euclidean neighbourhoods defined by a ball B(x,r) for each
        node x. 
    Args:
        mesh (numpy.ndarray): An array of nodes forming a mesh. 
        r (float): Radius of the desired ball neighbourhood.
        self_loops_allowed (boolean, optional): Boolean determining if nodes
                                                should have connection to themselves. 
                                                Defaults to False.
    Returns:
        Tuple: An adjacency matrix and distances between nodes
    """
    distance_mask, distances = generate_ball_neighbourhoods(mesh, r)

    if(not self_loops_allowed):
        np.fill_diagonal(distance_mask, False)

    sparse_adjacency = scipy.sparse.csr_matrix(distance_mask)
    sparse_distances = scipy.sparse.csr_matrix(distances[distance_mask])
    return sparse_adjacency, sparse_distances


def generate_2D_radial_mesh_equidistant(n_circles=10) -> Tuple(np.ndarray, np.ndarray):
    """Generates a mesh for a unit disk where nodes are arranged
        into consentric circles with equal arc length between them

    Args:
        n_circles (int, optional): Number of concentric circles. Defaults to 10.

    Returns:
        Tuple: The x and y coordinates for the nodes in numpy.ndarrays.
    """
    # Initalize node lists with origin.
    # Coordinates are in separate lists to make plotting easier
    x_coordinates = [0.]
    y_coordinates = [0.]

    # Define radii of the inner circles to be equidistant from eachother
    radii = np.linspace(1/n_circles, 1, n_circles)

    for i, r in enumerate(radii):
        # Calculate number of nodes in the currently iterated circle.
        # The angle used to determine this number is halved each iteration
        n_nodes = int(round(np.pi/np.arcsin(1/(2*(i+1)))))
        
        # Angles for each node distributed equally along the arc of a circle
        theta = np.linspace(0,2*np.pi,n_nodes)

        x_coordinates.extend(r*np.cos(theta[1:n_nodes+1]))
        y_coordinates.extend(r*np.sin(theta[1:n_nodes+1]))

        i += 1
    return np.array(x_coordinates), np.array(y_coordinates)



def generate_2D_radial_mesh_square_grid(n_nodes)-> Tuple(np.ndarray, np.ndarray):
    """Generates a mesh of a unit disk. Note that the method
        uses n_nodes to construct a square grid and discards
        coordinates outside the disk so the returned mesh
        contains floor(pi/4*n_nodes) nodes.

    Args:
        n_nodes (int): Number of nodes in the square grid [-1,1] X [-1,1]
                        that surrounds the unit disk.

    Returns:
        tuple: The x and y coordinates for the nodes in numpy.ndarrays.
    """
    # Generate the square mesh.
    x = np.linspace(-1,1,n_nodes)
    y = np.linspace(-1,1, n_nodes)
    xx, yy = np.meshgrid(x,y)
    
    # Define a mask for points within distance 1 from origin and filter the 
    # square mesh nodes with it.
    distance_mask = np.sqrt(xx**2+yy**2)<=1
    return xx[distance_mask], yy[distance_mask]


import matplotlib.pyplot as plt


x,y = generate_2D_radial_mesh_equidistant(5)
print(type(x))
plt.scatter(x,y)
plt.show()