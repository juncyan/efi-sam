import os
import cv2
import numpy as np
import pandas as pd
# from PIL import Image
import paddle
from paddle.io import Dataset
from .utils import one_hot_it


class DataReader(Dataset):
    def __init__(self, dataset_path, mode, load_edge = False, en_concat = True):
        
        self.data_dir = f"{dataset_path}/{mode}"

        self.load_edge = load_edge
        self.en_concat = en_concat

        self.data_list = self._get_list(self.data_dir)
        self.data_num = len(self.data_list)

        # if os.path.exists(os.path.join(dataset_path, 'label_info.csv')):
        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        
        assert self.data_list != None, "no data list could load"

        self.sst1_images = []
        self.gt_images = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(f"{self.data_dir}/image", _file))
            self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
                

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        lab_path = self.gt_images[index]

        A_img = cv2.imread(A_path)
        A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        A_img = self._normalize(A_img)#np.array(Image.open(A_path)))
        A_img = paddle.to_tensor(A_img.transpose(2, 0, 1)).astype('float32')
    
        label = cv2.imread(lab_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # label = np.array(Image.open(lab_path)) 

        label = one_hot_it(label, self.label_info)
        
        label = np.transpose(label, [2,0,1])
        label = paddle.to_tensor(label).astype('int64')
        data = {"img": A_img, "label": label}
        return data

    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path, 'image'))
        return data_list

    @staticmethod
    def _normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im

class TestReader(DataReader):
    def __init__(self, dataset_path, mode = "val", load_edge = False, en_concat = True):
        super(TestReader, self).__init__(dataset_path, mode, load_edge, en_concat)
        
        self.data_name = os.path.split(dataset_path)[-1]

        self.data_list = self._get_list(self.data_dir)
        self.data_num = len(self.data_list)
        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))

        self.file_name = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(f"{self.data_dir}/image", _file))
            self.gt_images.append(os.path.join(f"{self.data_dir}/label", _file))
            self.file_name.append(_file)

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        
        lab_path = self.gt_images[index]
        
        A_img = cv2.imread(A_path)
        # print(A_img)
        # assert A_img, "image load error, image is None"
        A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        A_img = self._normalize(A_img)#np.array(Image.open(A_path)))
        A_img = paddle.to_tensor(A_img.transpose(2, 0, 1)).astype('float32')
        label = cv2.imread(lab_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # A_img = self._normalize(np.array(Image.open(A_path)))
        # # w, h, _ = A_img.shape
        # A_img = paddle.to_tensor(A_img.transpose(2, 0, 1)).astype('float32')
      
        # label = np.array(Image.open(lab_path)) 
       
        label = one_hot_it(label, self.label_info)

        label = np.transpose(label, [2,0,1])
        label = paddle.to_tensor(label).astype('int64')

        data = {"img": A_img, "label": label, 'name': self.file_name[index]}
        return data
        


def detect_building_edge(data_path, save_pic_path):
    canny_low = 180
    canny_high = 210
    hough_threshold = 64
    hough_minLineLength = 16
    hough_maxLineGap = 3
    hough_rho = 1
    hough_theta = np.pi / 180
    image_names=os.listdir(data_path)
    for image_name in image_names:
        img=cv2.imread(os.path.join(data_path, image_name))
        shape=img.shape[:2]
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(img_gray,canny_low,canny_high)
        lines=cv2.HoughLinesP(edges,hough_rho,hough_theta,hough_threshold,hough_minLineLength,hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(os.path.join(save_pic_path, image_name),line_pic)
