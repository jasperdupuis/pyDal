"""

2d synthetic TL datasets.

This file shows how to train and validate using torch dataloader class, too.

TL models available:

Bellhop
Sine

Uniform + track based available.

Uses pickled storage mechanism from before.


"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# my modules

from synthetic_XY_class import Synthetic_f_xy_dataset
from synthetic_XY_class import create_synthetic_uniform_datasets

print("USING pytorch VERSION: ", torch.__version__)


FREQ = 50

# Define the hyperparameters
LEARNING_RATE= 1e-2
EPOCHS = 15
BATCH_SIZE = 250
NUM_WORKERS = 0 # 0 ==> main process, !0 ==> worker threads


class DeepNetwork(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''
    def __init__(self,):
        # Perform initialization of the pytorch superclass
        super(DeepNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n\nModel using {self.device} device, tensors to be moved in loops if needed")
        
        neural_net = nn.Sequential(
              nn.Linear(2, 256),
              nn.ReLU(),
              nn.Linear(256, 256),
              nn.ReLU(),
              nn.Linear(256, 256),
              nn.ReLU(),
              nn.Linear(256, 1),
              )
        self.neural_net = neural_net.to(self.device)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        x = self.neural_net(x) # see nn.Sequential
                
        return x

def train_batch( x, y ):
    """
    TRAIN A BATCH
    
    SHOULD BE ABLE TO PUT THIS IN MODEL CLASS ??? DOESNT WORK THO
    
    SO LEAVE OUTSIDE
    """
    # Before the backward pass,... ( DOING THIS FIRST) 
    # Use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)

    optimizer.zero_grad()

    # Run forward calculation        
    y_predict = model(x)
    
    # Compute loss.
    loss = loss_fn(y_predict, y)
    
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()
    
    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    return loss.data.item()




if __name__ == "__main__":        
    # mp.freeze_support()

    freq = FREQ
    freq_str = str(freq).zfill(4)
    fname = \
        'C:/Users/Jasper/Documents/Repo/pyDal/synthetic-data-sets/synthetic_TL/synthetic_TL_Bellhop_' \
            + freq_str \
            +'.pkl'
    
    dset_train = Synthetic_f_xy_dataset.build_dset_with_n_random_tracks_with_TL(
        fname,
        p_num_run = 20,
        p_std_angle = 2,
        p_std_SOG = 0.2,
        p_std_CPA = 6)

    dset_val = Synthetic_f_xy_dataset.build_dset_with_n_random_tracks_with_TL(
        fname,
        p_num_run = 4,
        p_std_angle = 2,
        p_std_SOG = 0.2,
        p_std_CPA = 6)

    # train data
    train_dataloader = DataLoader(
           dset_train, 
           batch_size=BATCH_SIZE,
           shuffle=True,
           num_workers = NUM_WORKERS,
           drop_last = True) # ignores last batch if not exactly the batch length

    # validation data
    validate_dataloader = DataLoader(
        dset_val , 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers = NUM_WORKERS,
        drop_last = True) # ignores last batch if not exactly the batch length
    
    # Instantiate the network (hardcoded in class)
    # model = DeepNetwork()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\nModel using {device} device, tensors to be moved in loops if needed")
        
    model = nn.Sequential(
              nn.Linear(2, 256),
              nn.ReLU(),
              nn.Linear(256, 256),
              nn.ReLU(),
              nn.Linear(256, 256),
              nn.ReLU(),
              nn.Linear(256, 1),
              )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()  # mean squared error
    
    epochs = EPOCHS
    loss_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    batch_index = 0
    
    # TRAINING
    for e in range(epochs):
        for i,data in enumerate(train_dataloader):
            x,y = data
            x = torch.reshape(x,(BATCH_SIZE,2))
            x = x.to(device)
            y = torch.reshape(y,(BATCH_SIZE,1))
            y = y.to(device)
            optimizer.zero_grad()

            # Run forward calculation        
            y_predict = model(x)
            
            # Compute loss.
            loss = loss_fn(y_predict, y)
            
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            loss = loss.data.item()
            optimizer.step()
            loss_train[e] = loss

            del x
            del y
            batch_index += 1
            
            
        # VALIDATION
        model.eval()
        temp_loss_list = list()
        for i,data in enumerate(validate_dataloader):
            xv,yv = data
            xv = xv.float()
            xv = torch.reshape(xv,(BATCH_SIZE,2))
            xv = xv.to(device)
            yv = torch.reshape(yv,(BATCH_SIZE,1))
            yv = yv.to(device)

            y_pred = model(xv)
            loss = loss_fn(input=y_pred, target=yv)
    
            temp_loss_list.append(loss.detach().cpu().numpy())
        
            del xv
            del yv
        
        loss_val[e] = np.average(temp_loss_list)

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)    
        print("\ttrain loss: %.7f" % loss_train[e])
        print("\tval loss: %.7f" % loss_val[e])
        print ("~~~ ~~~ ~~~")
           
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\nTest tensors are being moved to and from {device} device.\n\n")    
    
    # For results plotting we want to see the entire x, y surface, not just
    # the tracks. So : load the uniform data set.
    
    _, _, label, x_surface, y_surface = create_synthetic_uniform_datasets(fname)
    
    results = np.zeros_like(x_surface.flatten())
    index = 0
    xf = x_surface.flatten()
    yf = y_surface.flatten()
    for x,y in zip(xf,yf):
        test = torch.tensor((y,x))
        test = test.cuda()
        res = model(test.float())
        results[index] = res
        index = index + 1
        
    r = np.array(results)
    r[1]        

    result = np.reshape(results,x_surface.shape, order='C')
    delta = label-result
    
    label_zeromean = label-np.mean(label)
    result_zeromean = result - np.mean(result)
    
    x_track = np.array(dset_train.x)
    y_track = np.array(dset_train.y)
    
    vmin = -10 #db
    vmax = 10 #db
    
    x_min = np.min(x_surface)
    x_max = np.max(x_surface)
    y_min = np.min(y_surface)
    y_max = np.max(y_surface)
    extent = (x_min, x_max, y_min, y_max)
    
    cmap_string= 'Spectral'
    norm = matplotlib.colors.Normalize(vmin,vmax)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(17,8))
    fig.suptitle('Label, result, and delta plots \n Freq = ' + freq_str + ' Hz')
    ax1.imshow(
        label_zeromean,cmap=cmap_string,vmin=vmin,vmax=vmax,extent=extent,aspect='auto')
    ax1.title.set_text('True - mean(True)')
    ax2.imshow(
        result_zeromean,cmap=cmap_string,vmin=vmin,vmax=vmax,extent=extent,aspect='auto')
    ax2.title.set_text('Pred - mean(Pred)')
    ax3.imshow(
        delta,cmap=cmap_string,vmin=vmin,vmax=vmax,extent=extent,aspect='auto')
    ax3.title.set_text('Delta (True - Pred)')
    ax4.imshow(
        delta,cmap=cmap_string,vmin=vmin,vmax=vmax,extent=extent,aspect='auto')
    ax4.scatter(x_track,y_track,marker='.')
    ax4.title.set_text('Delta w/ training sample points')
    
    fig.colorbar(matplotlib.cm.ScalarMappable(norm,cmap_string),ax=[ax1,ax2,ax3,ax4])
    plt.show()
    
