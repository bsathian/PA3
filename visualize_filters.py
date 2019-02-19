import torch
import numpy as np
import matplotlib.pyplot as plt
from baseline_cnn import BasicCNN
from Arch2 import Arch2CNN
import sys



def visualize_weights(weights):
    n_cols = 3
    n_rows = np.ceil(len(weights)/n_cols)
    print(n_rows,n_cols)
    plt.figure(figsize = [3.2,2.4])
    weights = weights.detach().numpy()
    for i in range(len(weights)):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(weights[i][0])

    #plt.show()
    plt.savefig("weights.pdf")




#model= BasicCNN()
model = Arch2CNN()
model.load_state_dict(torch.load(sys.argv[1]))

visualize_weights(model.conv1.weight)

