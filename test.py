# encoding: utf-8
import torch
from rpi_define import *
import torch
from torch.utils.data import DataLoader
import  torchvision.datasets as dataset
import torchvision.transforms as transforms
from model import InceptionResNetV2
from glob import glob
import os
import cv2
import csv
from skimage import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def resize_test_images(src_image_dir, dst_image_dir):
    """
    将所有图片转换成相同的大小并存储到指定文件夹下
    :param src_image_dir: 原始图片所在的文件夹
    :param dst_image_dir: 存储到指定文件夹
    :return: None
    """
    src_images = glob(os.path.join(src_image_dir, "*.jpg"))
    for src_image_path in src_images:
        image_name = os.path.basename(src_image_path)
        dst_image_path = os.path.join(dst_image_dir, image_name)
        image_data = cv2.imread(src_image_path)
        output_image = cv2.resize(image_data, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite(dst_image_path, output_image)


def make_output_csv(src_image_dir, output_csv_path, model_path):
    """
    生成测试集的预测输出文件
    :param src_image_dir: 图片目录地址
    :param output_csv_path: 输出csv的文件地址
    :return: None
    """
    # 打开/创建输出的 csv 文件
    file_test = open(output_csv_path, 'a', newline='')
    csv_writer = csv.writer(file_test)
    csv_writer.writerow(["ImageName", "CategoryId"])

    # 枚举测试集图片
    image_paths = glob(os.path.join(src_image_dir, "test", "*.jpg"))
    print(len(image_paths))

    # 加载训练好的模型进行预测
    model = InceptionResNetV2()
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for idx, image_path in enumerate(image_paths):
        if idx %100==0:
            print("[%d/%d]"%(idx, len(image_paths)))
        image = io.imread(image_path)
        image_norm = transform(image)
        image_tensor = torch.unsqueeze(image_norm, 0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, 1)
        csv_writer.writerow([os.path.basename(image_path), predicted.item() + 1])



if __name__ == '__main__':
    # 测试集图片文件夹地址
    SRC_IMAGE_DIR = "/content/easy/"
    # 输出的结果文件 .csv 文件

    # 生成结果文件
    for i in range(8):
        model_name = "Classification_%d.pth"%(i+20)
        print(model_name)
        MODEL_PATH = os.path.join("./model",model_name)
        file_name = "output%d.csv"%(i+20)
        OUTPUT_CSV_PATH = os.path.join('./output', file_name)
        print(OUTPUT_CSV_PATH)
        make_output_csv(SRC_IMAGE_DIR, OUTPUT_CSV_PATH, MODEL_PATH)
