import eitnet
import matplotlib.pyplot as plt
import matplotlib as mpl
from graph import EIT_dataset
import utilities, random, data_loading

if __name__ == '__main__':
    model = eitnet.generate_EITNet()


    model.load_weights('weights/eit_checkp.index')
    data = EIT_dataset('fig_mats')

    ind = random.sample(range(len(data)), 10)

    data = data[ind]

    loader = utilities.WDJLoader(data[:1], batch_size = 1,node_level=True)

    mat_data = data_loading.load_data_from_mat("fig_mats/e20opad.mat")
    triang = mpl.tri.Triangulation(mat_data['nodes'][:,0],mat_data['nodes'][:,1], mat_data['tris'])
    
    print(mat_data['tris'].shape)
    pred = model.predict(loader.load(), steps=loader.steps_per_epoch)
    
    print(pred[:-20,:].shape)

    plt.tricontourf(triang, pred[:-20,:])
    plt.triplot(triang)
    plt.show()

    plt.tricontourf(triang, mat_data['volt_dist'][:,0])
    plt.triplot(triang)
    plt.show()

