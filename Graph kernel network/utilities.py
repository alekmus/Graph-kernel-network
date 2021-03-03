from typing import Tuple
import sklearn.metrics as skm
import spektral
import numpy as np
import scipy.spatial as sspa
from sklearn.metrics.pairwise import distance_metrics
import tensorflow as tf


def generate_ball_neighbourhoods(vectors, r) -> Tuple[np.ndarray, np.ndarray]:
    """Connects input vectors into neighbourhoods based on Euclidean distance.

    Args:
        vectors (numpy.ndarray): An array of Euclidean vectors.
        r (float): Radius of the ball used to determine neighbourhoods of vectors. 

    Returns:
        tuple: A dense boolean matrix that determines if the distance between two nodes
            is under r and a dense boolean matrix with actual distances.
    """
    distances = skm.pairwise_distances(vectors)
    distance_mask = distances <= r

    return distance_mask, distances


def generate_2D_radial_coordinates_equidistant(n_circles=10) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a mesh for a unit disk where nodes are arranged
        into consentric circles with equal arc length between them

    Args:
        n_circles (int, optional): Number of concentric circles. Defaults to 10.

    Returns:
        tuple: The x and y coordinates for the nodes in numpy.ndarrays.
    """
    # Initalize node lists with origin.
    # Coordinates are in separate lists to make plotting easier
    x_coordinates = [0.]
    y_coordinates = [0.]

    # Define radii of the circles to be equidistant from eachother
    radii = np.linspace(1/n_circles, 1, n_circles)

    for i, r in enumerate(radii):
        # Calculate number of nodes in the currently iterated circle.
        # The angle used to determine this number is halved each iteration
        n_nodes = int(round(np.pi/np.arcsin(1/(2*(i+1)))))
        
        # Angles for each node distributed equally along the arc of a circle
        theta = np.linspace(0,2*np.pi,n_nodes)

        x_coordinates.extend(np.round(r*np.cos(theta[1:n_nodes+1]),5))
        y_coordinates.extend(np.round(r*np.sin(theta[1:n_nodes+1]),5))

        i += 1
    return np.array(x_coordinates), np.array(y_coordinates)



def generate_2D_radial_coordinates_square_grid(n_nodes)-> Tuple[np.ndarray, np.ndarray]:
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
    distance_mask = np.sqrt(xx**2+yy**2)<1
    return xx[distance_mask], yy[distance_mask]


def insert_2D_disk_inclusion_to_coordinates(nodes, center_coordinate, r) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Splits a mesh into two parts by separating a ball B(center_coordinate, r)
        from the rest of the mesh.
    Args:
        mesh (tuple): Coordinates for the nodes in a mesh
        center_coordinate (tuple): Coordinates for the center of the disk inclusion
        r (float): Radius of an inclusion

    Returns:
        tuple: (x, y) coordinates for all nodes not in the inclusion and (x, y) coordinates in
                the inclusion.
    """
    x, y = nodes
    distance_mask = np.linalg.norm(((x-center_coordinate[0]), (y-center_coordinate[1])),axis=0) < r
    return (x[~distance_mask], y[~distance_mask]), (x[distance_mask], y[distance_mask])



def random_2D_disk_inclusion(nodes)-> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Creates a random 2D disk inclusion into mesh. The inclusion lies completely in the 
        interior of the mesh.

    Returns:
        tuple: (x, y) coordinates for all nodes not in the inclusion and (x, y) coordinates in
                the inclusion.
    """
    rand_x, rand_y, rand_r = np.random.rand(3)
    x,y = nodes
    # Choose closest actual node to the random vector to make sure the inclusion 
    # is centered around an actual node.
    distances = np.linalg.norm((x-rand_x, y-rand_y),axis=0)
    i = np.argmin(distances)

    while(np.linalg.norm((x[i],y[i]))+rand_r>=1):
        rand_x, rand_y, rand_r = np.random.rand(3)
        distances = np.linalg.norm((x-rand_x, y-rand_y),axis=0)
        i = np.argmin(distances)
    
    return insert_2D_disk_inclusion_to_coordinates(nodes, (x[i], y[i]), rand_r)


def centroids_from_tris(nodes, tris):
    """Computes centroids for triangles.

    Args:
        nodes (np.ndarray): Array of node coordinates
        tris (np.ndarray): Array of indices for triangles

    Returns:
        np.ndarray: Centroid coordinates for each triangle
    """
    return np.mean(nodes[tris],axis=1)


def mesh_from_nodes(nodes):
    """Creates a triangular mesh from points.

    Args:
        nodes (np.ndarray): Coordinates for nodes

    Returns:
        Scipy.spatial.qhull.Delaunay: Triangle mesh 
    """
    x, y = nodes
    return sspa.Delaunay(np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1), qhull_options='QJ')

# Need to clean up this loader. There seems to be a problem with 
# the library so workaround has to be made but I am scared to touch
# after this delayed the whole project for two weeks at least.
class WDJLoader(spektral.data.loaders.DisjointLoader):
    """Working disjoint loader
    """
    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [n_nodes, n_node_features]
        Edge features have shape [n_edges, n_edge_features]
        Targets have shape [..., n_labels]
        """
        signature = self.dataset.signature
        signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)
        
        output = []
        keys = ["x", "a", "e", "i"]
        for k in keys:
            if k in signature:
                shape = signature[k]["shape"]
                dtype = signature[k]["dtype"]
                spec = signature[k]["spec"]
                output.append(spec(shape, dtype))
        output = tuple(output)

        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]

        return (output, tf.TensorSpec((128*8,None), tf.float64))

if __name__ == '__main__':
    nodes = generate_2D_radial_coordinates_equidistant(5)
    x,y = nodes
    tris = mesh_from_nodes(nodes)
    print(tris.simplices)
    exit()
    import matplotlib.pyplot as plt 
    centroids = centroids_from_tris(tris)
    fig = plt.figure(figsize=(10,10))
    plt.triplot(x,y, tris.simplices)
    plt.scatter(centroids[:,0], centroids[:,1])
    plt.show()
  