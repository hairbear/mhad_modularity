# train.py
# Handong Wang 2021-07-20

# This has to be in the same directory as the overall Kinect folder

import os
import time
from rnn import RNNNet
import sys
import gym
import neurogym as ngym
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers import ScheduleEnvs

from PIL import Image
import cv2

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torchvision import transforms
from skimage import io

class KinectFeaturesDataset(dataset):

    def __init__(self, directory):
        self.directory = directory
        entries = os.listdir(directory)
        files = [entry for entry in entries if os.path.isfile(entry)]
        self.pgms = []
        self.pts = []
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension == ".pgm":
                self.pgms.append(file)
            elif extension == ".pt":
                self.pts.append(file)
        assert len(self.pgms) == len(self.pts) # Should be exactly 1 pytorch file corresponding to each pgm
        self.__len__ = len(self.pgms)
        self.pgms.sort()
        self.pts.sort()
        
        # load pts
        self.pgm_features = [torch.load(pt) for pt in self.pts]
        
        # do more stuff

    def __len__(self):
        self.__len__

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)

        return self.pgmfeatures[idx]

def train(kinect):
    pass

def train_local():
    pass

# def train_openmind(directory):
#     # can create all training sets in parallel just fine
#     kinect_dataset = KinectFeaturesDataset(directory)
#     dataloader = DataLoader(kinect_dataset, batch_size=4, shuffle=True, num_workers=4)


#     # not the same for training/backprop?
#     train(kinect)

def get_performance(net, env, num_trial=1000, device='cpu'):
    perf = 0

    for i in range(num_trial):

        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.DoubleTensor)
        action_pred, _ = net(inputs)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    return perf

def train_openmind(directory):
    # can create all training sets in parallel just fine
    kinect_dataset = KinectFeaturesDataset(directory)
    dataloader = DataLoader(kinect_dataset, batch_size=4, shuffle=True, num_workers=4)

    d = 7
    modelname = 'local_' + 'd_' + str(d) + '_model'
    kwargs = {'dt': 100}
    seq_len = 100

    tasks = ngym.get_collection('yang19')
    envs = [gym.make(task, **kwargs) for task in tasks]
    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
    dataset = ngym.Dataset(env, batch_size=4, seq_len=seq_len)
    env = dataset.env
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RNNNet(input_size=512, hidden_size=256, output_size=512, dt=env.dt, mask = None).to(device)
    
    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print_step = 200
    running_loss = 0.0
    running_task_time = 0
    running_train_time = 0

    losses = []
    perfs = []

    for i in range(40000):
        task_time_start = time.time()
        inputs, labels = dataset()

        # tasks = ngym.get_collection('yang19')
        # envs = [gym.make(task, **kwargs) for task in tasks]
        # schedule = RandomSchedule(len(envs))
        # env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        # act_size = env.action_space.n

        running_task_time += time.time() - task_time_start
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        train_time_start = time.time()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs, _ = model(inputs.double())
        #calculate loss
        loss = criterion(outputs.view(-1, act_size), labels)
        #do backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # # apply anatomical mask on h2h weights
        # model.rnn.h2h.weight.data = model.rnn.h2h.weight.data*(mask2d)
            
            
        running_train_time += time.time() - train_time_start
        #print statistics
        running_loss += loss.item()       
        if i % print_step == (print_step - 1):
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / print_step))
            losses.append(running_loss/print_step)
            running_loss = 0.0
            
            if True:
                print('Task/Train time {:0.1f}/{:0.1f} ms/step'.format(

                        running_task_time / print_step * 1e3,

                        running_train_time / print_step * 1e3))

                running_task_time, running_train_time = 0, 0
            perf = get_performance(model, env, num_trial=200, device=device)
            perfs.append(perf)
            print('{:d} perf: {:0.2f}'.format(i + 1, perf))
            fname = os.path.join('~/modular/neurogym/multitask', 'files', modelname + '.pt')
            torch.save(model.state_dict(), fname)

    print('Finished Training')
    losscurvename = '~/modular/neurogym/multitask/files/' + modelname + 'losscurve.npy'
    np.save(losscurvename,losses)
    testcurvename = '~/modular/neurogym/multitask/files/' + modelname + 'perfcurve.npy'
    np.save(testcurvename,perfs)

    # # not the same for training/backprop?
    # train(kinect)