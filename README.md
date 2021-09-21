Work in progress: the goal of this code is to reproduce the frame-by-frame motion of the Berkeley MHAD dataset, using variants on RNNs. Ultimately, the resulting RNNs will also be analyzed for signs of modularity.
Currently, the kinect_processing folder preprocesses the Berkeley MHAD dataset and most of the remaining code pertains to building and training the RNNs.

Dependencies: pytorch, torchvision, gym, neurogym, PIL, skimage, cv2

