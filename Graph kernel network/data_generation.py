import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import utilities

def read_forward_model(mat_file):
    """Reads an EIDORS forward model saved as a .mat from a given location and outputs 
       node coordinates, element triangles and indices of the electrode locations, respectively.
    Args:
        mat_file (dict): Matlab file accessed with scipy.io
    
    Returns:
        tuple: (Coordinates for nodes, triangle indices, electrode node indices)
    """
    
    nodes = mat_file['fmod'][0,0].nodes
    tris = mat_file['fmod'][0,0].elems-1

    #As many indices as there are electrodes [0,i]
    electrode_nodes = np.array([])
    for el in mat_file['fmod'][0,0].electrode[0]:
        electrode_nodes = np.append(electrode_nodes,el.nodes)
    electrode_nodes = electrode_nodes.astype(int)
    return nodes, tris, electrode_nodes, 


def compute_electrode_midpoints(all_nodes, electrode_nodes):
    """Computes the midpoints of electrode nodes to represent electrodes in the 
       GNN. CEM model in EIDORS uses two nodes to represent the electrodes.

    Args:
        all_nodes (np.array): Node locations in the mesh
        electrode_nodes (np.array): Indices for the electrode nodes

    Returns:
        np.ndarray: Coordinates for t
    """
    el_locs = all_nodes[electrode_nodes]
    
    el_locs = el_locs.reshape(-1,2,2)
    el_locs = np.sum(el_locs*0.5, axis=1)
    ex, ey = el_locs[:,0], el_locs[:,1]
    return ex, ey


def load_data_from_mat(file_location):
    """Loads simulation data from a matlab file

    Args:
        file_location (str): Location of a Matlab .mat file containing fields 'fmod', 'stim' and 'data'.
        'fmod' is an EIDORS forward model
        'stim' is an EIDORS stimulation object
        'data' contains the result of EIDORS fwd_solve function
    """
    f = sio.loadmat(file_location, struct_as_record=False)
    return read_forward_model(f)


def read_stimulation_patterns():
    pass
if __name__ == '__main__':
    
    nodes, tris, electrode_nodes = load_data_from_mat(r'data\data.mat')
    x = nodes[:, 0]
    y = nodes[:, 1]

    c = np.full(nodes.shape,'blue')
    c[electrode_nodes] = 'red'
    ex, ey = compute_electrode_midpoints(nodes, electrode_nodes)

    centroids = utilities.centroids_from_tris(nodes, tris)
    plt.figure(figsize=(10,10))
    plt.triplot(x,y,tris)
    plt.scatter(centroids[:,0],centroids[:,1],zorder =10)
    plt.scatter(x,y, c = c[:,0], zorder = 10)
    plt.scatter(ex,ey, c = 'm', zorder=20)
    plt.show()
