import eitnet
import matplotlib.pyplot as plt
import matplotlib as mpl
from graph import EIT_dataset
import utilities, random, data_loading
import networkx as nx
import numpy as np

def model_predictions():
    model = eitnet.generate_EITNet()


    model.load_weights('weights\\eit_checkp')
    data = EIT_dataset('fig_mats')

    ind = random.sample(range(len(data)), 10)

    data = data[ind]

    loader = utilities.WDJLoader(data[:2], batch_size = 1,node_level=True)
    # model.evaluate(loader.load(), steps=loader.steps_per_epoch)
    mat_data = data_loading.load_data_from_mat("mat_Data\\data1.mat")

    x = mat_data['nodes'][:,0]
    y = mat_data['nodes'][:,1]
    triang = mpl.tri.Triangulation(x,y, mat_data['tris'])
    
    pred = model.predict(loader.load(), steps=loader.steps_per_epoch)
    print(pred.shape)
    print(mat_data['volt_dist'][:,0].shape)
    plt.tricontourf(triang, pred[:x.shape[0]].flatten())
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
if __name__ == '__main__':
    #draw_process(r"fig_mats\data1.mat")
    model_predictions()
