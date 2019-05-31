# encoding: utf-8
import datetime
from rpi_define import *
from data_set import ImageDataSet
import numpy as np
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import argparse
from model import InceptionResNetV2
import time

parser = argparse.ArgumentParser("GAN learning")
parser.add_argument("--data-root", type=str, default="/content/easy",
                   help='data_root')
parser.add_argument('--batch-size', default=32, type=int,
                   help='batch_size(default:128)')
parser.add_argument('--num-epochs', default=20, type=int,
                   help='number of epoch')
parser.add_argument('--lr', default=0.0000001, type=float,
                   help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float,
                   help='beta hyperparam for Adam optimizers')
parser.add_argument('--model', default='./model',type=str, help="model_path")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def val():
#     model.eval()
#     loss =0
#     correct =0
#     with torch.no_grad():
#         for idx, val_data in enumerate(val_dataloader):
#             val_batch = val_data["image"].to(device)
#             val_label = val_data["label"].to(device)
#
#             output = model(val_batch)
#             loss += criterion(output, val_label.long()).item()
#
#             predict = torch.argmax(output, 1)
#             correct += torch.eq(predict, val_label).sum().item()
#     accuracy = correct / len(val_set)
#     return loss, accuracy

def train_model():

    model.train()
    print("start train")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(args.num_epochs):
        train_loss =0
        correct_total=0
        max_acc =0
        last_loss =0
        
        for i, sample_batch in enumerate(train_dataloader):
            print("epoch %d:\t [%d/%d]" %(epoch, i, len(train_dataloader)))
            data_batch  = sample_batch["image"].to(device)
            label_batch = sample_batch["label"].to(device)

            output = model(data_batch)
            loss = criterion(output, label_batch.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict= torch.argmax(output, 1)
            correct_total += torch.eq(predict, label_batch).sum().item()
            train_loss += loss.item()

        train_acc = correct_total / len(train_set)

        if train_acc > max_acc or train_loss < last_loss:
            print("save model at epoch {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(args.model, "Classification_{}.pth").format(epoch+24))
            max_acc = train_acc
            last_loss = train_loss
            print("max_acc: %.5f\t last_loss: %.5f"%(max_acc, last_loss))

        # val_loss, val_acc = val()
        f = open("out.txt", "a")
        print("[%d/%d]\t train_loss:%.5f\t train_acc:%.5f\t "%
              (epoch, args.num_epochs, train_loss, train_acc), file=f)
        f.close()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    model = InceptionResNetV2()
    if torch.cuda.device_count() >1:
        print("let us use", torch.cuda.device_count(), " GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("./model/Classification_23.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 训练集
    train_data_path = os.path.join(args.data_root, "data")
    train_csv_path = os.path.join(args.data_root, "data.csv")
    # val_csv_path = os.path.join(args.data_root, "val.csv")


    train_set = ImageDataSet(train_data_path, train_csv_path,
                             transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                             ]))

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # 验证集
    # val_set = ImageDataSet(train_data_path, val_csv_path,
    #                          transform=transforms.Compose([
    #                              transforms.ToPILImage(),
    #                              transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    #                          ]))
    # val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    if not os.path.exists(args.model):
        os.makedirs(args.model)

    #开始训练模型
    train_model()

