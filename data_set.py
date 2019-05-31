# encoding: utf-8

import os
import random
from rpi_define import *
import pandas as pd
from skimage import io
import csv
import cv2
import torch
from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, images_dir, csv_file_path, transform=None):
        self.csv_data = pd.read_csv(csv_file_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, item):
        image_name = os.path.join(self.images_dir, self.csv_data.iloc[item, 0])
        image = io.imread(image_name)

        lebal = self.csv_data.iloc[item, 1]
        sample ={"image": image, "label": lebal-1}
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample




def split_train_and_val_data_set(src_csv_path, train_csv_path, val_csv_path):
    """
    从原始数据集中分离出训练集和验证集
    :param src_csv_path: 原始数据集标注.csv文件地址
    :param train_csv_path: 生成的训练集数据标注.csv文件地址
    :param val_csv_path: 生成的验证集数据标注.csv文件地址
    :return: None
    """
    file_train = open(train_csv_path, 'w', newline='')
    train_writer = csv.writer(file_train)
    train_writer.writerow(["ImageName", "CategoryId"])

    file_val = open(val_csv_path, 'w', newline='')
    val_writer = csv.writer(file_val)
    val_writer.writerow(["ImageName", "CategoryId"])

    csv_data = pd.read_csv(src_csv_path)

    categories = set()
    for i in range(csv_data.shape[0]):
        category = csv_data["CategoryId"][i]
        if (category not in categories) or (random.randint(0, 9) > 0):
            train_writer.writerow([csv_data["ImageName"][i], category])
            categories.add(category)
        else:
            val_writer.writerow([csv_data["ImageName"][i], category])


# def convert_images_to_same_size(csv_path, src_image_dir, dst_image_dir):
#     """
#     将所有图片转换成相同的大小并存储到指定文件夹下
#     :param csv_path: 标注数据的.csv文件地址
#     :param src_image_dir: 原始图片所在的文件夹
#     :param dst_image_dir: 存储到指定文件夹
#     :return: None
#     """
#     csv_data = pd.read_csv(csv_path)
#     for image_name in csv_data["ImageName"]:
#         src_image_path = os.path.join(src_image_dir, image_name)
#         dst_image_path = os.path.join(dst_image_dir, image_name)
#         img = cv2.imread(src_image_path)
#         output = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
#         cv2.imwrite(dst_image_path, output)


# if __name__ == '__main__':
    # 原始数据集
    # SRC_IMAGE_DIR = "/content/easy/data/"
    # SRC_CSV_PATH = "/content/easy/data.csv"

    # 处理成统一大小图片的数据集
#     NORM_IMAGE_DIR = "/home/zhouxiaolun/PycharmProjects/NJU_competition/easy/norm_data/"
#     TRAIN_CSV_PATH = "/content/easy/train.csv"
    # VAL_CSV_PATH = "/content/easy/val.csv"

    # 将所有图片转换成相同的大小
#     if not os.path.exists(NORM_IMAGE_DIR):
#         os.makedirs(NORM_IMAGE_DIR)
#         convert_images_to_same_size(SRC_CSV_PATH, SRC_IMAGE_DIR, NORM_IMAGE_DIR)

    # 从原始数据集中生成训练集和验证集
    # split_train_and_val_data_set(SRC_CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH)

