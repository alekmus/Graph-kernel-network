from typing import Tuple
import sklearn.metrics as skm
import scipy.sparse
import numpy as np
import scipy.spatial as sspa
from sklearn.metrics.pairwise import distance_metrics
# TODO THis might actually be more difficult than implementing the ndmap myself

# Decomposed geometry matrix for the unit disc
# required for datageneration with matlab
UNIT_DISC_DGM = np.array([
    [ 1, 1, 1, 1], # circle domain
    [-1, 0, 1, 0], # Starting x-coordinate of boundary segment
    [ 0, 1, 0,-1], # Ending x-coordinate of boundary segment
    [ 0,-1, 0, 1], # Starting y-coordinate of boundary segment
    [-1, 0, 1, 0], # Ending y-coordinate of boundary segment
    [ 1, 1, 1, 1], # Left minimal region label
    [ 0, 0, 0, 0], # Right minimal region label
    [ 0, 0, 0, 0], # x-coordinate of the center of the circle
    [ 0, 0, 0, 0], # y-coordinate of the center of the circle
    [ 1, 1, 1, 1], # radius of the circle
    [ 0, 0, 0, 0], # dummy row to match up with the ellipse geometry 
    [ 0, 0, 0, 0] # dummy row to match up with the ellipse geometry 
    ])


def generate_matlab_pet_matrices(tris):
    """Generates mesh data triple [point matrix, edge matrix, triangle matrix]
       that matlab uses to solve pdes.

    Args:
        tris (scipy.spatial.qhull.Delaunay object): Triangle data
    """
    p = tris.T.points

