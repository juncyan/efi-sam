import numpy as np
import cv2
import tifffile as tf
import os
import glob
from skimage import io
from PIL import Image

# img1 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\after\after.tif"
# label1 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\after\after_label.tif"
# img2 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\before\before.tif"
# label2 = r"D:\Datasets\Building change detection dataset\1. The two-period image data\before\before_label.tif"
# clab = r"D:\Datasets\Building change detection dataset\1. The two-period image data\change label\change_label.tif"



src_path = r"/mnt/data/Datasets/HRSCD/src/{}/{}"
dst_path = r"/mnt/data/Datasets/HRSCD/train/{}/{}"
TARGET_SIZE = 256

def crop_images(file:str,idx:int, src_path, dst_path):

    im1 = io.imread(src_path.format("A",file))
    file_ = file.replace("-2005-", "-2012-")
    file_ = file_.replace("-2006-", "-2012-")
    file_ = file_.split(".")[0]
    
    fb = glob.glob(src_path.format("B", f"{file_}*"))[0]
    f1 = fb.replace("/B/", "/labelA/")
    f2 = fb.replace("/B/", "/labelB/")
    f3 = fb.replace("/B/", "/label/")
    # print(fb, f1, f2, f3)
    im2 = io.imread(fb)
    lab1 = io.imread(f1)
    lab2 = io.imread(f2)
    label = io.imread(f3)
    # print(im1.shape, im2.shape, lab1.shape, lab2.shape, label.shape)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    # lab1 = cv2.cvtColor(lab1)
    # lab2 = cv2.cvtColor(lab2)
    # label = cv2.cvtColor(label)

    TARGET_SIZE = 512
    width, height, _ = im1.shape
    nw = int(width / TARGET_SIZE)
    nh = int(height / TARGET_SIZE)
    width = TARGET_SIZE * nw
    height = TARGET_SIZE * nh

    ws = np.linspace(0, width, nw + 1, dtype=np.int32)
    width_idx1 = ws[0: -1]
    width_idx2 = ws[1:]
    hs = np.linspace(0, height, nh + 1, dtype=np.int32)
    height_idx1 = hs[0: -1]
    height_idx2 = hs[1:]
    f = file.split(".")[0]
    print(f, label.shape)
    for x in range(len(width_idx1)):
        for y in range(len(height_idx1)):
            file_name = "{}_{}_{}.png".format(idx, x, y)
            img1 = im1[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]
            img2 = im2[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y], :]
            la1 = lab1[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]
            la2 = lab2[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]
            la = label[width_idx1[x]: width_idx2[x], height_idx1[y]: height_idx2[y]]

            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
            la1 = Image.fromarray(la1)
            la2 = Image.fromarray(la2)
            la = Image.fromarray(la)

            img1.save(dst_path.format('A', file_name))
            img2.save(dst_path.format('B', file_name))
            la1.save(dst_path.format('labelA', file_name))
            la2.save(dst_path.format('labelB', file_name))
            la.save(dst_path.format('label', file_name))


if __name__ == "__main__":
    print("split")
    src_dir = r"/mnt/data/Datasets/HRSCD/src/A"
    fl = open(r"/mnt/data/Datasets/HRSCD/filename_idx_list.txt", "w+")
    fils = os.listdir(src_dir)
    # print(fils)
    for idx, f in enumerate(fils):
        fl.write(f"{idx}_is_{f}\n")
        crop_images(f, idx, src_path, dst_path)
    fl.close()