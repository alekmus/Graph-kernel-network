import eitnet
import matplotlib.pyplot as plt
import matplotlib as mpl
from graph import EIT_dataset
import utilities, random, data_loading

if __name__ == '__main__':
    model = eitnet.generate_EITNet()


    model.load_weights(r'C:\Users\Aleksi\Documents\hy-opinnot\Gradu\Gradu_code\Graph-kernel-network\Graph kernel network\weights\eit_checkp')
    data = EIT_dataset('fig_mats')

    ind = random.sample(range(len(data)), 10)

    data = data[ind]

    loader = utilities.WDJLoader(data, batch_size = 1,node_level=True)
    model.evaluate(loader.load(), steps=loader.steps_per_epoch)
    mat_data = data_loading.load_data_from_mat(r"C:\Users\Aleksi\Documents\hy-opinnot\Gradu\Gradu_code\Graph-kernel-network\Graph kernel network\fig_mats\e20opad.mat")
    #triang = mpl.tri.Triangulation(mat_data['nodes'][:,0],mat_data['nodes'][:,1], mat_data['tris'])
    
    print( mat_data['volt_dist'][:,0].shape)
    pred = model.predict(loader.load(), steps=loader.steps_per_epoch)
    
    plt.tripcolor(mat_data['nodes'][:,0],mat_data['nodes'][:,1], mat_data['tris'], pred[:-20].flatten())
    plt.show()
