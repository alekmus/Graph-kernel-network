from numpy.core.numeric import zeros_like
import spektral
import utilities
import data_loading
import scipy.sparse
import numpy as np
import os

class EIT_dataset(spektral.data.Dataset):
    def __init__(self, mat_data_dir, **kwargs):
        """
        Args:
            mat_data_dir (str): Name of the directory that contains the .mat files.
                                Needs to be located in the same directory as the script.
        """
        self.mat_data = mat_data_dir
        super().__init__(**kwargs)

    @property
    def path(self):
        """Defines a property that contains a path to the graph data folder
        Returns:
            str: Path to the graph data folder
        """
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Should not exist')
            
    def read(self):
        """ Reads the files in the data set and returns a list of graphs.
        Returns:
            list: List of spektral.data.Graph objects
        """
        graphs = []
        mat_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.mat_data)
      

        for f in os.listdir(mat_data_dir):
            path = os.path.join(mat_data_dir,f)
            graph_factory = mat_graph_factory(path)
            graphs.extend(graph_factory.generate_graphs())
            
        return graphs

class mat_graph(spektral.data.Graph):
    """Object that converts EIDORS forward model data from a .mat file to a spektral graph.
    """
    def __init__(self, node_coords,
                 adjacency_matrix, 
                 edge_features, 
                 electrode_coords,
                 stimulation_pattern, 
                 measurement_pattern,  
                 conductivity,
                 volts,
                 tris):

        # Stimulation current is constant across samples so it is not implemented for now.
        # It could however be added by simply as a node feature.
        
        node_features = self.construct_node_features(node_coords, electrode_coords, stimulation_pattern, conductivity, measurement_pattern)
        target = volts[tris]
        target = np.concatenate([target, np.zeros((electrode_coords.shape[0],3))])
        super().__init__(x = node_features, a = adjacency_matrix, e = edge_features, y = target)


    def construct_node_features(self, node_coords, electrode_coords, stimulation_pattern, conductivity, measurement_pattern):
        """Constructs node feature vectors for each node in the graph.
           Feature vectors contain the following: [x: float, y: float, conductivity: float, in: [0,1], out: [0,1]]
           Conductivity at electrode nodes is set to 1.
        Args:
            node_coords (np.ndarray): Coordinates for ccentroids of triangle elements.
            electrode_coords (np.ndarray): Coordinates for the electrode midpoints.
            stimulation_pattern (np.array): Stimulation pattern.
            measurement_pattern (np.array): Measurement pattern
        Returns:
            np.ndarray: Feature vector for the graph nodes.
        """
        feats = node_coords
        
        feats = np.concatenate([feats, conductivity.reshape(-1,1)],axis=1)
        feats = np.concatenate([feats, np.zeros((feats.shape[0],4))],axis=1)
        
        el_vectors = electrode_coords
        el_vectors = np.concatenate([el_vectors, np.ones((el_vectors.shape[0],1))], axis=1)
        el_vectors = np.concatenate([el_vectors, (stimulation_pattern>0).reshape(-1,1)], axis=1)
        el_vectors = np.concatenate([el_vectors, (stimulation_pattern<0).reshape(-1,1)], axis=1)
        
        el_vectors = np.concatenate([el_vectors, (measurement_pattern>0).reshape(-1,1)], axis=1)
        el_vectors = np.concatenate([el_vectors, (measurement_pattern<0).reshape(-1,1)], axis=1)
        

        return np.concatenate((feats, el_vectors),axis=0)


    def construct_targets(self, node_coords, volts):
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
        target = np.concatenate([target, electrode_targets], axis=0)
        target = target.reshape(-1,1)
        return target


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
        self.tris = data['tris']
        self.nodes = utilities.centroids_from_tris(coords, data['tris'])
        
        self.volts = data['volt_dist']
        self.electrode_coords = data_loading.compute_electrode_midpoints(coords, self.electrode_indices)
        self.adj, self.edge_feats = self.build_connections(np.concatenate([self.nodes, self.electrode_coords],axis=0))
        
        
    
    def generate_graphs(self):
        """Generates all samples from a single .mat file by iterating over 
           all the measurement and stimulation patterns given by it.
        Returns:
            list: Graphs  
        """
        graphs = []
        for stim_p,meas_ps, volt in zip(self.stim_patterns,self.meas_patterns, self.volts.T):
            for meas_p in meas_ps:
                graphs.append(mat_graph(self.nodes, 
                                        self.adj, 
                                        self.edge_feats, 
                                        self.electrode_coords, 
                                        stim_p, 
                                        meas_p, 
                                        self.conductivity,
                                        volt,
                                        self.tris))
                    

        return graphs

    def build_connections(self, nodes, r=0.1, self_loops_allowed=True):
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

        return scipy.sparse.csr_matrix(distance_mask.astype(int)), distances[distance_mask].reshape(-1,1)

if __name__ == "__main__":
    print(EIT_dataset('mat_data'))