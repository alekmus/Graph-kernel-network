import eitnet
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from graph import EIT_dataset
import utilities, random, data_loading
import networkx as nx
import numpy as np
import cv2
from mayavi import mlab

def model_predictions():
    model = eitnet.generate_EITNet()


    model.load_weights('weights\\EITNet_weights_020321')
    data = EIT_dataset('fig_mats')

    i = 1

    loader = utilities.WDJLoader(data[i:i+1], batch_size = 1,node_level=True)
    # model.evaluate(loader.load(), steps=loader.steps_per_epoch)
    mat_data = data_loading.load_data_from_mat("fig_mats\\data1.mat")

    x = mat_data['nodes'][:,0]
    y = mat_data['nodes'][:,1]
    triang = mpl.tri.Triangulation(x,y, mat_data['tris'])
    
    pred = model.predict(loader.load(), steps=loader.steps_per_epoch)
    plt.subplot(122, aspect='equal')
    plt.tricontourf(triang, pred[:x.shape[0]].flatten())

    plt.subplot(121, aspect='equal')
    plt.tricontourf(triang, mat_data['volt_dist'][:,i])
    plt.show()

def draw_process(mat_file):
    mat_data = data_loading.load_data_from_mat(mat_file)
    x,y = mat_data['nodes'][:,0], mat_data['nodes'][:,1]
    cond = mat_data['conductivity']
    cent = utilities.centroids_from_tris(mat_data['nodes'], mat_data['tris'])
    nodes = np.concatenate((mat_data['nodes'],cent),axis = 0)
    connections = utilities.generate_ball_neighbourhoods(nodes, r=0.15)[0]
    plt.subplot(131, aspect = 'equal')
    plt.title('FEM mesh')
    plt.axis('off')
    plt.tripcolor(x, y, mat_data['tris'], facecolors=cond, edgecolors='lightsteelblue',cmap='Blues')
    plt.scatter(x,y, s=1)

    plt.subplot(132, aspect = 'equal')
    plt.title('FEM mesh with element centroids')
    plt.axis('off')
    plt.tripcolor(x, y, mat_data['tris'], facecolors=cond, edgecolors='lightsteelblue',cmap='Blues')
    x = np.concatenate((x,cent[:,0]),axis=0)
    y = np.concatenate((y,cent[:,1]),axis=0)
    plt.scatter(x,y, s=1)

    plt.subplot(133, aspect = 'equal')
    plt.title('Final graph structure with r=0.3')
    G = nx.from_numpy_array(connections)
    #plt.tripcolor(x, y, mat_data['tris'], facecolors=cond, edgecolors='lightsteelblue',cmap='Blues')
    nx.draw(G, pos=nodes, node_size=2, edgecolors='k')
    

    plt.show()

def block_diag(*arrs):
    
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def draw_gkn(mat_file):
    mat_data = data_loading.load_data_from_mat(mat_file)
    x,y = mat_data['nodes'][:,0], mat_data['nodes'][:,1]
    cent = utilities.centroids_from_tris(mat_data['nodes'], mat_data['tris'])
    volt = mat_data['volt_dist'][:,0]
    cond = mat_data['conductivity']
    cond = np.concatenate([np.ones(x.shape[0]),cond])
    
    
    hidden = (np.random.rand(*cond.shape)*2-1)*0.4
    out = np.concatenate([volt+1, np.ones(cent.shape[0])])
    cond = np.concatenate([cond, hidden, out], axis=0)
    cond /= np.max(cond)
    cond = np.log10(cond+1)

    nodes = np.concatenate((mat_data['nodes'],cent),axis = 0)
    pos1 = np.concatenate((nodes, np.zeros((nodes.shape[0],1))), axis=1)
    pos2 = np.concatenate((nodes, np.ones((nodes.shape[0],1))), axis=1)
    pos3 = np.concatenate((nodes, 2*np.ones((nodes.shape[0],1))), axis=1)
    

    pos = np.concatenate([pos1,pos2,pos3], axis=0)
    connections = utilities.generate_ball_neighbourhoods(nodes, r=0.1)[0]
    connections = block_diag(connections, connections, connections)
    G = nx.from_numpy_array(connections)
    pts = mlab.points3d(pos[:,0], pos[:,1], pos[:,2], scale_factor=0.05, scale_mode="none", colormap='coolwarm')
    pts.mlab_source.dataset.point_data.scalars = cond

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.005)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()

if __name__ == '__main__':
    #draw_process(r"fig_mats\data1.mat")
    model_predictions()
    #draw_gkn(r"fig_mats\data1.mat")
