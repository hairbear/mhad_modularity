# kinect_processing_parallel.py
# Handong Wang 2021-07-06

# This has to be in the same directory as the overall Kinect folder


import os
import sys

from PIL import Image
import cv2

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Ignore the color photos: we only want to replicate the outlines, 
# for which the pgms are sufficient
def color_process(image_file:str):
    resnet18 = models.resnet18(pretrained=True)
    print("ppm", image_file)
    pass

# Extract features from the pgms to save into pt files
def pgm_process(image_file:str, background, write=False):

    print("Reading for pgm processing", image_file)

    # To extract features, we use resnet18 minus the fully connected layer
    # This does almost all of the feature extraction of resnet
    # But gives a result of a 512-long feature vector instead of a softmax vector
    resnet18 = models.resnet18(pretrained=True)
    resnet18_features = torch.nn.Sequential(*list(resnet18.children())[:-1])

    image = cv2.imread(image_file)
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_background.png"), background * 8)
    # retval, image = cv2.threshold(image, 14, 27, cv2.THRESH_TRUNC)
    fg_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Replace missing foreground pixel values with background pixel values
    fg_gray[fg_gray == 0] = background[fg_gray == 0]
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_foreground.png"), fg_gray * 8)

    # Difference isolates the foreground-only content: i.e. human outline
    dif_gray = cv2.absdiff(background, fg_gray)
    # Crop out some unnecessary background noise
    dif_gray = dif_gray[0:480, 0:480]
    # Arbitrary scaling up
    dif_gray *= 8
    # Threshold as final filter
    retval, dif_gray = cv2.threshold(dif_gray, 255, 255, cv2.THRESH_TRUNC)
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_dif.pgm"), dif_gray)

    # Convert to color, input into resnet
    dif = cv2.cvtColor(dif_gray, cv2.COLOR_GRAY2BGR)
    transform = transforms.Compose([               #[1]
	    transforms.Resize(224),                    #[2]
	    transforms.ToTensor(),                     #[3]
	    transforms.Normalize(                      #[4]
	    mean = [0.485, 0.456, 0.406],              #[5]
	    std = [0.229, 0.224, 0.225])])             #[6]
    
    foreground_t = transform(Image.fromarray(dif))
    batch_t = torch.unsqueeze(foreground_t, 0)
    resnet18_features.eval()
    out = resnet18_features(batch_t)

    # Save output and finish
    torch.save(out, image_file.replace(".pgm", "_features.pt"))
    print("Processed pgm:", image_file)

    return out

# String replacement approach to determine
# the corresponding background file from an image file
def get_background_file(image_file):
    # Background files are located within S01 only
    image_file = image_file.replace("s02", "s01")
    image_file = image_file.replace("S02", "S01")

    # A08 and less involve no chairs, background B01
    image_file = image_file.replace("a01", "b01")
    image_file = image_file.replace("a02", "b01")
    image_file = image_file.replace("a03", "b01")
    image_file = image_file.replace("a04", "b01")
    image_file = image_file.replace("a05", "b01")
    image_file = image_file.replace("a06", "b01")
    image_file = image_file.replace("a07", "b01")
    image_file = image_file.replace("a08", "b01")
    # A09 and above use a chair, background B02
    image_file = image_file.replace("a09", "b02")
    image_file = image_file.replace("a10", "b02")
    image_file = image_file.replace("a11", "b02")


    image_file = image_file.replace("A01", "B01")
    image_file = image_file.replace("A02", "B01")
    image_file = image_file.replace("A03", "B01")
    image_file = image_file.replace("A04", "B01")
    image_file = image_file.replace("A05", "B01")
    image_file = image_file.replace("A06", "B01")
    image_file = image_file.replace("A07", "B01")
    image_file = image_file.replace("A08", "B01")
    image_file = image_file.replace("A09", "B02")
    image_file = image_file.replace("A10", "B02")
    image_file = image_file.replace("A11", "B02")

    # remove the rep count
    image_file = image_file.replace("R01/", "")
    image_file = image_file.replace("R02/", "")
    image_file = image_file.replace("R03/", "")
    image_file = image_file.replace("R04/", "")
    image_file = image_file.replace("R05/", "")
    image_file = image_file.replace("r01_", "")
    image_file = image_file.replace("r02_", "")
    image_file = image_file.replace("r03_", "")
    image_file = image_file.replace("r04_", "")
    image_file = image_file.replace("r05_", "")

    image_file = image_file[:-9] + "00000.pgm"

    return image_file

# Fill in zero-valued pixels from the background
# With the average value of the neighbor pixels
# Slow, but produces a reasonable gradient
def fill_background(image_file, background):
    while not np.all(background):
        print("Filling in background pixels:", np.count_nonzero(background), "of", background.size, "pixels complete")
        new_background = np.copy(background)
        for i in range(len(background)):
            for j in range(len(background[0])):
                if background[i, j] < 5:
                    new_background[i, j] = neighbor_avg(background, i, j, 7)
        background = new_background
        # cv2.imwrite(image_file.replace(".pgm", "_background.png"), background * 8)
    return background

def neighbor_avg(background, i, j, k):
    count = 0
    sum = 0
    left = max(0, i - k)
    right = min(len(background), i + k)
    top = max(0, j - k)
    bottom = min(len(background[0]), j + k)
    for x in range(left, right):
        for y in range(top, bottom):
            if (x - i) ** 2 + (y - j) ** 2 > k * k:
                continue
            sum += background[x, y]
            if background[x, y] > 2:
                count += 1
            # count += 1
    if count == 0:
        return 0
    return sum // count

def get_background_image(image_file):
    print("Getting background image", image_file)
    background = cv2.imread(get_background_file(image_file))
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    bg_gray = fill_background(image_file, bg_gray)
    return bg_gray

def main(id:int):    
    # Convert from 1-indexed to 0-indexed
    id -= 1

    # Get the correct directory from the id number
    kinect_set = os.getcwd() + "/Kinect"

    kinects = ["/Kin01", "/Kin02"]
    subjects = ["/S01", "/S02", "/S03", "/S04", "/S05", "/S06", "/S07", "/S08", "/S09", "/S10", "/S11", "/S12"]
    actions = ["/A01", "/A02", "/A03", "/A04", "/A05", "/A06", "/A07", "/A08"]
    reps = ["/R01", "/R02", "/R03", "/R04", "/R05"]

    rep = reps[id % 5]
    id //= 5
    action = actions[id % 8]
    id //= 8
    subject = subjects[id % 12]
    id //= 12
    kinect = kinects[id % 2]
    background = get_background_image(kinect_set + kinect + "/S01/B01/kin_k01_s01_b01_depth_00000.pgm")

    target = kinect_set + kinect + subject + action + rep

    # For local debugging

    # pgm_process("kin_k01_s01_a01_r01_depth_00164.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00165.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00166.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00167.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00168.pgm", background_kin01, True)
    # pgm_process("kin_k02_s01_a01_r01_depth_00000.pgm", background_kin02, True)
    # pgm_process("kin_k01_s07_a06_r02_depth_00188.pgm", background_kin01, True)

    print("Beginning processing of directory" + target)
    for root, dirs, files in os.walk(target):
        for file in files:
            filename, extension = os.path.splitext(file)
            # match extension:
            #     case ".ppm":
            #         color_process(root + file)
            #     case ".pgm":
            #         pgm_process(root + file)
            #     case ".pt":
            #         pass
            #     case _:
            #         print("Bad file extension for file:", root + "/" + file)
            print("Reading file", file)
            if extension == ".ppm":
                color_process(root + "/" + file)
            elif extension == ".pgm":
                pgm_process(root + "/" + file, background)
            elif extension == ".pt":
                pass
            else:
                print("Bad file extension for file:", root, file, filename, extension)
    print("Kinect analysis complete.")

if __name__ == "__main__":
    print(sys.argv, sys.argv[1])
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        print("No argument, nothing done")