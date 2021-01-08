import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import utilities

def read_geometry(mat_file):
    """Reads an EIDORS forward model from a .mat file and outputs 
       node coordinates, element triangles and indices of the electrode locations, respectively.
    Args:
        mat_file (dict): Matlab .mat file accessed with scipy.io
    
    Returns:
        tuple: (Coordinates for nodes, triangle indices, electrode node indices)
    """
    
    nodes = mat_file['fmod'][0,0].nodes
    tris = mat_file['fmod'][0,0].elems-1

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
    # Find coordinates of the electrode nodes
    el_locs = all_nodes[electrode_nodes]
    el_locs = el_locs.reshape(-1,2,2)

    # CEM electrodes in EIDORS consist of two boundary nodes and the connected edge.
    # Construct a new node located in the middle of the edge that represent this construct.
    el_locs = np.sum(el_locs*0.5, axis=1)
    return el_locs


def load_data_from_mat(file_location):
    """Loads simulation data from a matlab file

    Args:
        file_location (str): Location of a Matlab .mat file containing fields 'fmod', 'stim' and 'data'.
        'fmod' is an EIDORS forward model
        'stim' is an EIDORS stimulation object
        'data' contains the result of EIDORS fwd_solve function
        'img' contains and EIDORS image object with conductivity distribution information
    Returns:
        dict: Dictionary {"nodes":node locations,
                          "tris": indices of triangle endpoints, 
                          "electrode_nodes": indices of electrode nodes,
                          "stim_pattern": stimulation patterns,
                          "meas_pattern": measurement patterns, 
                          "measurements": measurements at electrodes for each measurement pattern, 
                          "conductivity": conductivity distribution within the object}
    """
    mat_file = sio.loadmat(file_location, struct_as_record=False)
    stim_pattern, meas_pattern = read_patterns(mat_file)
    nodes, tris, electrode_nodes = read_geometry(mat_file)

    # Read measurements given by electrodes.
    # Measurements are given in a single array whereas the indices
    # of the measurement patterns are split based on possible configurations 
    meas = read_electrode_measurements(mat_file)
    conductivity = load_conductivity(mat_file)
    return {"nodes":nodes,
            "tris": tris,
            "electrode_nodes": electrode_nodes,
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "measurements": meas,
            "conductivity": conductivity}


def load_conductivity(mat_file):
    return mat_file['img'][0,0].elem_data.astype(float).flatten()


def read_electrode_measurements(mat_file):
    """Reads measurement data from a Matlab file

    Args:
        mat_file (dict): Matlab .mat file accessed with scipy.io

    Returns:
        np.ndarray: Measurement data
    """
    return mat_file['data'][0,0].meas


def read_patterns(mat_file):
    """Reads an EIDORS stimulation and measurement pattern saved as a .mat from a given location.
    Args:
        mat_file (dict): Matlab .mat file accessed with scipy.io
    
    Returns:
        np.ndarray: Stimulation patterns
    """
    stim_patterns = []
    meas_patterns = []

    # Read patterns and convert them to dense representation
    # Dense matrices might not be what we want, but we'll see
    for stim in mat_file['stim'][0]:
        stim_patterns.append(stim.stim_pattern.toarray().flatten())
        meas_patterns.append(stim.meas_pattern.toarray())

    # Convert lists to numpy arrays before returning for easier manipulation
    return np.array(stim_patterns), np.array(meas_patterns)


if __name__ == '__main__':
    d  = load_data_from_mat(r'data\data.mat')
    nodes =d['nodes']
    tris = d['tris'] 
    electrode_nodes = d['electrode_nodes']
    stim_pattern = d['stim_pattern']
    meas_pattern = d['meas_pattern'] 
    meas = d['measurements'] 
    conductivity = d['conductivity']
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    c = np.full(nodes.shape,'blue')
    c[electrode_nodes] = 'red'
    el_locs = compute_electrode_midpoints(nodes, electrode_nodes)
    ex, ey = el_locs[:,0], el_locs[:,1]
    centroids = utilities.centroids_from_tris(nodes, tris)

    
    print(conductivity.shape)
    print(centroids.shape)
    print(electrode_nodes.shape)
    
    
    plt.figure(figsize=(10,10))
    #plt.triplot(x,y,tris)
    #plt.scatter(x,y, zorder = 10)

    #plt.scatter(x,y, c = 'white', zorder = 1)
    plt.scatter(ex,ey, c = 'm', zorder=20)
    
    plt.tripcolor(x,y,tris, facecolors = conductivity, edgecolors ='k',cmap='Greys')
    plt.scatter(centroids[:,0],centroids[:,1],zorder =10)
   
    #plt.scatter(centroids[:,0],centroids[:,1],c=conductivity, zorder =10, cmap='Greys')
    midpoint_nodes = np.vstack([ex,ey]).T

    centroids = np.append(centroids,midpoint_nodes,axis=0)

    """
    conn,_ = utilities.generate_ball_neighbourhoods(centroids,0.3)

    for cs in conn:
        n = centroids[cs]
        for i in range(len(n)):
            for j in range(i+1, len(n)):
                plt.plot([n[i,0],n[j,0]],[n[i,1],n[j,1]], c='k')
    """
    plt.show()
