# resnet_rnn.py
# Handong Wang 2021-07-20

# This has to be in the same directory as the overall Kinect folder

# Basic libraries
import os
import sys
import time

# Image processing libraries
from PIL import Image
import cv2
from skimage import io

# Machine learning libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torchvision import transforms

# gym and neurogym modules
import gym
import neurogym as ngym
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers import ScheduleEnvs

# Local imports
from eval import eval_local
from rnn import RNNNet
from masks import Mask
from kinect_dataset import KinectFeaturesDataset, KinectPgmsDataset

torch.set_default_tensor_type(torch.DoubleTensor)

# Local training (not implemented)
# def train(kinect):
#     pass

def train_openmind(directory, use_mask, steps, print_step):

    # For each slurm job, create a new kinect dataset for this directory
    kinect_dataset = KinectFeaturesDataset(directory)

    mask = None
    if use_mask:
        directory += "mask_"
        mask = Mask.mask2d(32, 16, cutoff=3, periodic=False)

    # Set fundamental parameters
    lr = 1e-3 # learning rate
    # d = 7
    # modelname = 'model_' + directory
    # kwargs = {'dt': 100}
    # seq_len = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RNNNet(input_size=512, hidden_size=512, output_size=512, dt=None, mask=mask).to(device)
    rnn = model.rnn

    # Train the model over a single iteration of the directory
    def train_iter():
        hidden = rnn.init_hidden(device)
        rnn.zero_grad()

        iter_loss = 0
        
        criterion = nn.MSELoss()

        for i in range(len(kinect_dataset) - 1):
            input = torch.reshape(kinect_dataset[i], (1, -1))
            input.to(device)
            outputs, hidden = model(input.double())
            ans = torch.reshape(kinect_dataset[i + 1], (1, -1)).double()
            loss = criterion(outputs, ans)
            iter_loss += loss.item()
            loss.backward()
            for p in rnn.parameters():
                p.data.add_(p.grad.data, alpha=-lr)

        return outputs, iter_loss / (len(kinect_dataset) - 1) # return average loss per pgm

    optimizer = torch.optim.Adam(model.parameters(), lr)
    running_loss = 0.0
    running_task_time = 0
    running_train_time = 0

    losses = []
    perfs = []

    # Train the model over $steps iteration of the directory
    for i in range(steps):
        train_time_start = time.time()
        
        # zero the parameter gradients
        # optimizer.zero_grad()
        
        # forward + backward + optimize
        output, loss_item = train_iter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: I guess? I'm not sure which parameters specifically
        optimizer.step()
        
        # apply anatomical mask on h2h weights
        # TODO: Apply this???? Seems kind of essential to maintain local connectivity
        # model.rnn.h2h.weight.data = model.rnn.h2h.weight.data*(mask2d)
            
            
        running_train_time += time.time() - train_time_start
        #print statistics
        running_loss += loss_item   
        if i % print_step == (print_step - 1):
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / print_step))
            losses.append(running_loss/print_step)
            running_loss = 0.0
            
            if True:
                print('Task/Train time {:0.1f}/{:0.1f} ms/step'.format(

                        running_task_time / print_step * 1e3,

                        running_train_time / print_step * 1e3))

                running_task_time, running_train_time = 0, 0
            # fname = os.path.join('/Users/handongwang/modular/neurogym/multitask', 'files', str(i // print_step) + '.pt')
            fname = directory + 'model_state_dict_' + str(i // print_step) + '.pt'
            torch.save(model.state_dict(), fname)

    losscurvename = directory + 'losscurve.npy'
    print('Finished training. Saved to', losscurvename)
    np.save(losscurvename,losses)

def eval_openmind(model_directory, directories, use_mask, steps, print_step):

    mask = None
    if use_mask:
        model_directory += "mask_"
        mask = Mask.mask2d(32, 16, cutoff=3, periodic=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fname = model_directory + 'model_state_dict_' + str((steps // print_step) - 1) + '.pt'
    model = RNNNet(input_size=512, hidden_size=512, output_size=512, dt=None, mask = mask).to(device)
    model.load_state_dict(torch.load(fname))
    model.eval()
    rnn = model.rnn

    # Evaluate the model over a single iteration of the directory
    def eval_iter():

        iter_loss = 0
        final_loss = 0

        criterion = nn.MSELoss()

        for i in range(len(kinect_dataset) - 1):
            input = torch.reshape(kinect_dataset[i], (1, -1))
            input.to(device)
            outputs, hidden = model(input.double())
            ans = torch.reshape(kinect_dataset[i + 1], (1, -1)).double()
            loss = criterion(outputs, ans)
            lossitem = loss.item()
            iter_loss += lossitem
            if i == len(kinect_dataset) - 2:
                final_loss = lossitem

        return outputs, iter_loss / (len(kinect_dataset) - 1), final_loss

    print_step = 200
    running_loss = 0.0
    running_final_loss = 0.0

    losses = []
    final_losses = []

    # Evaluate the model over $steps iteration of the directory
    for i in range(len(directories)):
        directory = directories[i]

        kinect_dataset = KinectFeaturesDataset(directory)
        
        # forward + backward + optimize
        output, loss_item, final_loss_item = eval_iter()

        # print statistics
        running_loss += loss_item   
        running_final_loss += final_loss_item
        losses.append(loss_item)
        final_losses.append(final_loss_item)

    print('Finished evaluating with an avg loss of', running_loss/len(directories), 'and an avg final loss of', running_final_loss/len(directories))
    eval_losses = model_directory + 'eval_losses.npy'
    np.save(eval_losses,losses)
    eval_final_losses = model_directory + 'eval_final_losses.npy'
    np.save(eval_final_losses,final_losses)

def get_directory_from_id_openmind(root, id:int):
    kinect_set = root + "/Kinect"
    kinects = ["/Kin01", "/Kin02"]
    subjects = ["/S01", "/S02", "/S03", "/S04", "/S05", "/S06", "/S07", "/S08", "/S09", "/S10", "/S11", "/S12"]
    actions = ["/A01", "/A02", "/A03", "/A04", "/A05", "/A06", "/A07", "/A08"]
    reps = ["/R01", "/R02", "/R03", "/R04", "/R05"]

    n_reps = len(reps)          # 5
    n_actions = len(actions)    # 8
    n_subjects = len(subjects)  # 12
    n_kinects = len(kinects)    # 2

    rep = reps[id % n_reps]
    id //= n_reps
    action = actions[id % n_actions]
    id //= n_actions
    subject = subjects[id % n_subjects]
    id //= n_subjects
    kinect = kinects[id % n_kinects]
    id //= n_kinects
    assert id == 0  # This confirms that the original value of id was within range

    return kinect_set + kinect + subject + action + rep + "/"

def train_local():
    pass

def main_local():
    train_local()
    eval_local()

def main_openmind(id:int, train:bool, eval:bool, mask:bool):
    directory = get_directory_from_id_openmind(os.getcwd(), id)
    steps = 20000
    print_step = 200
    # directory = "/Users/handongwang/modular/pgms/"
    print("directory", directory)
    if train:
        print("Beginning Kinect openmind training.")
        train_openmind(directory, mask, steps, print_step)
    else:
        print("Skipping train.")
    if eval:
        print("Beginning Kinect openmind eval.")
        directories = [get_directory_from_id_openmind('/om/user/handong', i) for i in range(100)]
        eval_openmind(directory, directories, mask, steps, print_step)
    else:
        print("Skipping eval.")
    print("Kinect openmind complete.")

if __name__ == "__main__":
    # Because my local machine testset and the full dataset on openmind
    # have different directory structures

    local = False
    train = False
    eval = True
    use_mask = True
    
    if local:
        main_local()
    else:
        print(sys.argv, sys.argv[1])
        if len(sys.argv) > 1:
            main_openmind(int(sys.argv[1]), train, eval, use_mask)
        else:
            print("No argument, nothing done")
        # main_openmind()