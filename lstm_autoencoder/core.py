# Standard Library
import os, sys, random
import pandas as pd
import numpy as np
from pathlib import Path

# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

# Local Modules
import autoencoder

###############
# GPU Setting #
###############
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # comment this line if you want to use all of your GPUs
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


####################
# Data preparation #
####################
def prepare_dataset(sequential_data):
    if type(sequential_data) == pd.DataFrame:
        data_in_numpy = np.array(sequential_data)
        data_in_tensor = torch.tensor(data_in_numpy, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(0)
    elif type(sequential_data) == np.ndarray:
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(0)
    elif type(sequential_data) == list:
        data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(0)
        
    seq_len = unsqueezed_data.shape[1]
    no_features = unsqueezed_data.shape[2] 
    # shape[0] is the number of batches
    
    return unsqueezed_data, seq_len, no_features


##################################################
# QuickEncode : Encoding & Decoding & Final_loss #
##################################################
def QuickEncode(input_data, 
                embedding_dim, 
                id, 
                learning_rate = 1e-3, 
                every_epoch_print = 100, 
                epochs = 10000, 
                patience = 100, 
                max_grad_norm = 0.005):
    
    refined_input_data, seq_len, no_features = prepare_dataset(input_data)
    print("refined", refined_input_data.shape, seq_len, no_features)
    model = autoencoder.LSTM_AE(seq_len, no_features, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm)
    final_loss = model.fit(refined_input_data, id)
    
    # recording_results
    embedded_points = model.encode(refined_input_data)
    decoded_points = model.decode(embedded_points)

    return embedded_points.cpu().data, decoded_points.cpu().data, final_loss

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

def get_seqs(directory, seq_len):
    entries = os.listdir(directory)
    # print("entries", entries)
    files = [directory + entry for entry in entries if os.path.isfile(directory + entry)]
    # print("files", files)
    pgms = []
    pts = []
    for file in files:
        # print("file", file)
        filename, extension = os.path.splitext(file)
        if extension == ".pgm":
            pgms.append(file)
        elif extension == ".pt" and "features" in filename:
            pts.append(file)
    # print("phase 2")
    # print("Kinect Dataset Length", len(pgms), len(pts))
    assert len(pgms) == len(pts) # Should be exactly 1 pytorch file corresponding to each pgm
    # print("Kinect Dataset Length", len(self.pts))
    pgms.sort()
    pts.sort()
    
    # load and flatten pts
    pgm_features = [torch.load(pt).detach().numpy().flatten() for pt in pts]
    return np.array(pgm_features[:seq_len])

# def get_input_data(directory_nums, seq_len, num_features):
#     num_seqs = len(directory_nums)
#     input_data = np.empty((num_seqs, seq_len, num_features))
#     for i in range(num_seqs):
#         # parent of lstm_autoencoder is /om/user/handong/
#         # thus, use the full dataset
#         directory = get_directory_from_id_openmind(str(path.parent.absolute()), directory_nums[i])
#         input_data[i,:,:] = get_seqs(directory, seq_len)
#     print("input", type(input_data), input_data.shape)
#     return input_data

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         path = Path(os.getcwd())
#         num_directories = int(sys.argv[1])
#         directory_nums = random.sample([i for i in range(480)], num_directories)
#         print("Training off of directories", directory_nums)
#         num_seqs = num_directories
#         seq_len = 100 # Each directory safely has at least 100 pgms
#         num_features = 512 # The pgm pts all contain 512 numbers
#         input_data = get_input_data(directory_nums, seq_len, num_features)
#         embedding_dim = 512 # To make it as comparable to the RNN approach as possible, match 512
#         QuickEncode(input_data, embedding_dim)
#     else:
#         print("No argument, nothing done")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(os.getcwd())
        id = int(sys.argv[1])
        directory = get_directory_from_id_openmind(str(path.parent.absolute()), id)
        seq_len = 100 # Each directory safely has at least 100 pgms
        num_features = 512 # The pgm pts all contain 512 numbers
        input_data = get_seqs(directory, seq_len)
        embedding_dim = 512 # To make it as comparable to the RNN approach as possible, match 512
        print("Beginning training...")
        embedded_points_data, decoded_points_data, final_loss = QuickEncode(input_data, embedding_dim, id)
        print("Finished training.", type(embedded_points_data), type(decoded_points_data))
        torch.save(embedded_points_data, os.getcwd() + "/embedded_points_" + str(id) + ".pt")
        torch.save(decoded_points_data, os.getcwd() + "/decoded_points_" + str(id) + ".pt")
        print("Saved.")
        with open('losses.txt', 'a') as f:
            print('Final loss for directory', str(id) + ":", final_loss, file=f)
    else:
        print("No argument, nothing done")