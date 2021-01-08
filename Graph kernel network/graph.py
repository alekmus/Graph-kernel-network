from scipy.sparse import construct
import spektral
import utilities
import data_loading
from typing import Tuple
import scipy.sparse
import numpy as np


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
            node_coords ([type]): [description]
            electrode_coords ([type]): [description]
            stimulation_pattern ([type]): [description]

        Returns:
            [type]: [description]
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
        """[summary]

        Args:
            node_coords ([type]): [description]
            measurement_pattern ([type]): [description]
            measurement ([type]): [description]

        Returns:
            [type]: [description]
        """
        target = np.zeros((node_coords.shape[0],3))
        electrode_targets = np.zeros((measurement_pattern.shape[0],3))
        
        electrode_targets[:,1][measurement_pattern>0] = 1
        electrode_targets[:,2][measurement_pattern<0] = 1
        electrode_targets[:,0][(measurement_pattern<0) | (measurement_pattern>0)] = measurement
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

        mat_graph(self.nodes, 
                  self.adj, 
                  self.edge_feats, 
                  self.electrode_coords, 
                  self.stim_patterns[0], 
                  self.meas_patterns[0][0],
                  self.meas[0], 
                  self.conductivity)
    

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
    mat_graph_factory(r'data\data.mat')