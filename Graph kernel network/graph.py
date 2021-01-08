import spektral
import utilities
import data_loading
from typing import Tuple
import scipy.sparse
import numpy as np
from pathlib import Path
import os

class EIT_dataset(spektral.data.Dataset):
    def __init__(self, **kwargs):
        path = os.path.dirname(os.path.realpath(__file__))
        spektral.datasets.utils.DATASET_FOLDER = path
        print(self.path)
        super().__init__(**kwargs)
    def download(self):
        pass

    def path(self):
        return os.path.join('C:Aleksi', self.__class__.__name__)


class mat_graph(spektral.data.Graph):
    def __init__(self, node_coords,
                 adjacency_matrix, 
                 edge_features, 
                 electrode_coords,
                 stimulation_pattern, 
                 measurement_pattern, 
                 measurement, 
                 conductivity):

        # Stimulation current is constant across samples so it is not implemented for now.
        # It could however be added by simply as a node feature.
        node_features = self.construct_node_features(node_coords, electrode_coords, stimulation_pattern, conductivity)
        target_measurements = self.construct_targets(node_coords, measurement_pattern, measurement)
        super().__init__(x = node_features, a = adjacency_matrix, e = edge_features, y = target_measurements)


    def construct_node_features(self, node_coords, electrode_coords, stimulation_pattern, conductivity):
        """Constructs node feature vectors for each node in the graph.
           Feature vectors contain the following: [x: float, y: float, conductivity: float, in: [0,1], out: [0,1]]
           Conductivity at electrode nodes is set to 1.
        Args:
            node_coords (np.ndarray): Coordinates for ccentroids of triangle elements.
            electrode_coords ([np.ndarray): Coordinates for the electrode midpoints.
            stimulation_pattern (np.array): Stimulation pattern.

        Returns:
            np.ndarray: Feature vector for the graph nodes.
        """
        feats = node_coords
        feats = np.concatenate([feats, conductivity.reshape(-1,1)],axis=1)
        feats = np.concatenate([feats, np.zeros((feats.shape[0],2))],axis=1)
        
        el_vectors = electrode_coords
        el_vectors = np.concatenate([el_vectors, np.ones((el_vectors.shape[0],1))], axis=1)
        el_vectors = np.concatenate([el_vectors, (stimulation_pattern>0).reshape(-1,1)], axis=1)
        el_vectors = np.concatenate([el_vectors, (stimulation_pattern<0).reshape(-1,1)], axis=1)
        
        return np.concatenate((feats, el_vectors),axis=0)


    def construct_targets(self, node_coords, measurement_pattern, measurement):
        """Constructs target vector for the nodes. Zero elsewhere except measurement nodes.

        Args:
            node_coords (np.ndarray): Coordinates for the nodes. Only used to get their number. Possible site for optimization.
            measurement_pattern (np.array): Single measurement pattern
            measurement (float): The voltage measured at the pertinent nodes.

        Returns:
            np.array: Targets for each node.
        """
        target = np.zeros((node_coords.shape[0]))
        electrode_targets = np.zeros((measurement_pattern.shape))
        
        electrode_targets[(measurement_pattern<0) | (measurement_pattern>0)] = measurement
        return np.concatenate([target, electrode_targets], axis=0)


class mat_graph_factory():
    def __init__(self, mat_file_location):
        """
        Args:
            mat_file (str): file_location (str): Location of a Matlab .mat file containing fields 'fmod', 'stim' and 'data'.
                            'fmod' is an EIDORS forward model
                            'stim' is an EIDORS stimulation object
                            'data' contains the result of EIDORS fwd_solve function
                            'img' contains and EIDORS image object with conductivity distribution information
        """
        data  = data_loading.load_data_from_mat(mat_file_location)
        coords = data['nodes']
        self.stim_patterns = data['stim_pattern']
        self.meas_patterns = data['meas_pattern'] 
        self.meas = data['measurements'] 
        self.conductivity = data['conductivity']
        self.electrode_indices = data['electrode_nodes']
        self.nodes = utilities.centroids_from_tris(coords, data['tris'])
        self.electrode_coords = data_loading.compute_electrode_midpoints(coords, self.electrode_indices)
        self.adj, self.edge_feats = self.build_connections(np.concatenate([self.nodes, self.electrode_coords],axis=0))

        
    
    def generate_graphs(self):
        """Generates all samples from a single .mat file by iterating over 
           all the measurement and stimulation patterns given by it.
        Returns:
            list: Graphs  
        """
        graphs = []
        meas_idx = 0
        for i, stim_p in enumerate(self.stim_patterns):
            for meas_p in self.meas_patterns[i]:
                graphs.append(mat_graph(self.nodes, 
                                        self.adj, 
                                        self.edge_feats, 
                                        self.electrode_coords, 
                                        stim_p, 
                                        meas_p,
                                        self.meas[meas_idx], 
                                        self.conductivity))
                meas_idx += 1

        return graphs

    def build_connections(self, nodes, r=0.1, self_loops_allowed=False) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
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
            tuple: An adjacency matrix and distances between nodes
        """
        distance_mask, distances = utilities.generate_ball_neighbourhoods(nodes, r)

        if(not self_loops_allowed):
            np.fill_diagonal(distance_mask, False)

        sparse_adjacency = scipy.sparse.csr_matrix(distance_mask)
        sparse_distances = scipy.sparse.csr_matrix(distances[distance_mask])
        return sparse_adjacency, sparse_distances

if __name__ == "__main__":
    a= EIT_dataset()