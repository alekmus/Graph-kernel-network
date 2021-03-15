import eitnet
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from graph import EIT_dataset
import utilities, random, data_loading
import networkx as nx
import numpy as np

def model_predictions():
    model = eitnet.generate_EITNet()


    model.load_weights('weights\\norm_eit_checkp')
    data = EIT_dataset('fig_mats')
    

    i = 7

    loader = utilities.WDJLoader(data[i:i+1], batch_size = 1,node_level=True)
    #model.evaluate(loader.load(), steps=loader.steps_per_epoch)
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
    plt.tripcolor(x, y, mat_data['tris'], facecolors=cond, edgecolors='lightsteelblue',cmap='Blues')
    nx.draw(G, pos=nodes, node_size=2, edgecolors='k')
    

    plt.show()

def draw_gkn():
    x   = np.linspace(1,5,100)
    y1  = np.ones(x.size)
    y2  = np.ones(x.size)*2
    y3  = np.ones(x.size)*3
    z   = np.sin(x/2)
    pl.figure()
    ax = pl.subplot(projection='3d')
    ax.plot(x, y1, z, color='r')
    ax.plot(x, y2, z, color='g')
    ax.plot(x, y3, z, color='b')

    ax.add_collection3d(pl.fill_between(0.95*x, z, 1.05*z, color='r', alpha=0.3), zs=1, zdir='z')
    ax.add_collection3d(pl.fill_between(0.90*x, z, 1.10*z, color='g', alpha=0.3), zs=2, zdir='z')
    ax.add_collection3d(pl.fill_between(0.85*x, z, 1.15*z, color='b', alpha=0.3), zs=3, zdir='z')

    ax.set_xlabel('Day')
    ax.set_zlabel('Resistance (%)')
    pl.show()

if __name__ == '__main__':
    #draw_process(r"fig_mats\data1.mat")
    model_predictions()
    #draw_gkn()
