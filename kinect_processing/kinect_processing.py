# Handong Wang 2021-07-06

# This has to be in the same directory as the overall Kinect folder

import os

from PIL import Image
import cv2

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def color_process(image_file:str):
    resnet18 = models.resnet18(pretrained=True)
    print("ppm", image_file)
    pass

def pgm_process(image_file:str, background, write=False):

    print("Reading for pgm processing", image_file)

    # This model is pretrained on a color image - this is a grayscale model
    resnet18 = models.resnet18(pretrained=True)
    resnet18_features = torch.nn.Sequential(*list(resnet18.children())[:-1])

    image = cv2.imread(image_file)
    # background = cv2.imread(image_file)
    # background = cv2.imread(get_background_file(image_file))
    # background = get_background_image(image_file)

    # print(np.amax(image), np.amax(background), np.amin(image), np.amin(background))
    # print(background.shape)

    # background with depth filter
    # retval, background = cv2.threshold(background, 0, 27, cv2.THRESH_TRUNC)
    # bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    # bg_gray = fill_background(image_file, bg_gray)
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_background.png"), background * 8)
    # retval, image = cv2.threshold(image, 14, 27, cv2.THRESH_TRUNC)
    fg_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fg_gray[fg_gray == 0] = background[fg_gray == 0]
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_foreground.png"), fg_gray * 8)

    # print("image", image)
    # print("background", get_background_file(image_file))
    # print("shapes", image.shape, background.shape)

    dif_gray = cv2.absdiff(background, fg_gray)
    # Crop out some unnecessary background noise:
    dif_gray = dif_gray[0:480, 0:480]
    # Arbitrary scaling up
    dif_gray *= 8
    retval, dif_gray = cv2.threshold(dif_gray, 255, 255, cv2.THRESH_TRUNC)
    # dif_gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
    # dif = cv2.cvtColor(dif_gray, cv2.COLOR_GRAY2BGR)
    if write:
        cv2.imwrite(image_file.replace(".pgm", "_dif.pgm"), dif_gray)
    dif = cv2.cvtColor(dif_gray, cv2.COLOR_GRAY2BGR)

    """
    # Attempt to cut out the floor
    # Abandoned since it cuts out the legs first
    # retval, dif_thresh = cv2.threshold(dif, 96, 255, cv2.THRESH_TOZERO)
    # dif_thresh_gray = cv2.cvtColor(dif_thresh, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(image_file.replace(".pgm", "_dif_thresh.pgm"), dif_thresh_gray)

    # Mask approach doesn't actually end up working, too much noise
    # in the Kinect data
    # retval, foreground_mask = cv2.threshold(background - image, 1, 255, cv2.THRESH_BINARY) # basis of a mask on image
    # foreground_mask_gray = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("foreground_mask.pgm", foreground_mask_gray)
    # foreground_mask = cv2.bitwise_not(foreground_mask)
    # foreground = cv2.bitwise_and(image, foreground_mask)
    # foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("output.pgm", foreground_gray)
    """

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

    torch.save(out, image_file.replace(".pgm", "_features.pt"))

    print("Processed pgm:", image_file)

    return out

# A very dumb but easy to understand approach
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

def main():    
    kinect_set = os.getcwd() + "/Kinect"
    kinect1 = kinect_set + "/Kin01"
    kinect2 = kinect_set + "/Kin02"
    background_kin01 = get_background_image(kinect1 + "/S01/B01/kin_k01_s01_b01_depth_00000.pgm")
    background_kin02 = get_background_image(kinect2 + "/S01/B01/kin_k02_s01_b01_depth_00000.pgm")
    # background_kin01 = get_background_image("kin_k01_s01_b01_depth_00000.pgm")
    # background_kin02 = get_background_image("kin_k02_s01_b01_depth_00000.pgm")

    # pgm_process("kin_k01_s01_a01_r01_depth_00164.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00165.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00166.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00167.pgm", background_kin01, True)
    # pgm_process("kin_k01_s01_a01_r01_depth_00168.pgm", background_kin01, True)
    # pgm_process("kin_k02_s01_a01_r01_depth_00000.pgm", background_kin02, True)
    # pgm_process("kin_k01_s07_a06_r02_depth_00188.pgm", background_kin01, True)

    print("Beginning Kinect 1 processing.")
    for root, dirs, files in os.walk(kinect1):
        # Don't analyze background data
        if "B0" in root:
            continue
        if "A09" in root or "A10" in root or "A11" in root:
            continue
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
                pgm_process(root + "/" + file, background_kin01)
            elif extension == ".pt":
                pass
            else:
                print("Bad file extension for file:", root, file, filename, extension)
    print("Beginning Kinect 2 processing.")
    for root, dirs, files in os.walk(kinect2):
        # Don't analyze background data
        if "B0" in root:
            continue
        if "A09" in root or "A10" in root or "A11" in root:
            continue
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
                pgm_process(root + "/" + file, background_kin02)
            elif extension == ".pt":
                pass
            else:
                print("Bad file extension for file:", root, file, filename, extension)
    print("Kinect analysis complete.")

if __name__ == "__main__":
    main()