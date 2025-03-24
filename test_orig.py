import transformers
import pandas as pd
from astropy.io import fits
import time
import math
import random
import matplotlib.pyplot as plt
#import healpy as hp
from astropy.io import fits
import matplotlib.image as mp
from astropy.coordinates import SkyCoord
from collections import Counter
import os
from astropy.io import ascii
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from multiprocessing import Pool
from PIL import Image
import sys

import numpy as np  
import torch  
import cv2  
import torch.nn as nn  
from transformers import ViTModel, ViTConfig  
from torchvision import transforms  
from torch.optim import Adam  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
  
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

#Pretrained model checkpoint  
model_checkpoint = '/home/quhan/ML_try/dataset/vit-base-patch16-224-in21k'  

import GPUtil

# 获取 GPU 使用信息
gpus = GPUtil.getGPUs()

for gpu in gpus:
    print("------------------------------------")
    print(f"GPU ID: {gpu.id}")
    print(f"名称: {gpu.name}")
    print(f"显存使用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
    print(f"GPU 使用率: {gpu.load * 100:.2f}%")
    print(f"温度: {gpu.temperature}°C")



class ImageDataset(torch.utils.data.Dataset):  
  
    def __init__(self, input_data):  

        self.input_data = input_data  
        # Transform input data  
        self.transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Resize((224, 224), antialias=True),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  
        std=[0.5, 0.5, 0.5])  
        ])  

    def __len__(self):  
        return len(self.input_data)  

    def get_images(self, idx):  
        return self.transform(self.input_data[idx]['image'])  

    def get_labels(self, idx):  
        return self.input_data[idx]['label']  

    def __getitem__(self, idx):  
        # Get input data in a batch  
        train_images = self.get_images(idx)  
        train_labels = self.get_labels(idx)  

        return train_images, train_labels

class ViT(nn.Module):  
  
    def __init__(self, config=ViTConfig(), num_labels=2,  
        model_checkpoint= '/home/quhan/ML_try/dataset/vit-base-patch16-224-in21k'):  
        super(ViT, self).__init__()  

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)  
        self.classifier = (  
        nn.Linear(config.hidden_size, num_labels)  
        )  

    def forward(self, x):  

        x = self.vit(x)['last_hidden_state']  
        # Use the embedding of [CLS] token  
        output = self.classifier(x[:, 0, :])  

        return output
def model_train(dataset, epochs, learning_rate, bs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load nodel, loss function, and optimizer
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load batch image
    train_dataset = ImageDataset(dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=bs, shuffle=True)

    # Fine tuning loop
    for i in range(epochs):
        total_acc_train = 0
        total_loss_train = 0.0

        for train_image, train_label in tqdm(train_dataloader):
            output = model(train_image.to(device))
            loss = criterion(output, train_label.to(device))
            acc = (output.argmax(dim=1) == train_label.to(device)).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epochs: {i + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')
        save_path="result_model/SL_model_test_ephoch"+str(i)+".model"
        torch.save(model.state_dict(), save_path)
        localtime = time.asctime( time.localtime(time.time()) )
        times=localtime.split()[3]        
        print(localtime)
    return model

def predict(img,the_model):  
    use_cuda = torch.cuda.is_available()  
    device = torch.device("cuda" if use_cuda else "cpu")  
    transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Resize((224, 224)),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])  
    img = transform(img)  
    output = the_model(img.unsqueeze(0).to(device))  
    
    prediction = output.argmax(dim=1).item()  
    print(output)
    return(output,prediction)
def batch_predict(device,images, model, batch_size=64):
    #use_cuda = torch.cuda.is_available()  
    #device = torch.device("cuda" if use_cuda else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    images = [transform(img) for img in images]
    dataloader = DataLoader(images, batch_size=batch_size)

    results = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            #predictions = output.argmax(dim=1)
            #print(output)
            results.extend(output.cpu().clone().detach().numpy())

    return results

from datasets import load_dataset  
from datetime import datetime
#time.sleep(25000)    

## 超参数设置
EPOCHS = 50
LEARNING_RATE = 5e-5 
BATCH_SIZE = 40

print("\nEPOCHS =",EPOCHS)
print("LEARNING_RATE =",LEARNING_RATE)
print("BATCH_SIZE =",BATCH_SIZE)
print("\n   ")


import torch



#lrl=[3e-6,5e-6,1e-5,1e-6]  
#lrl=[3e-6,5e-6,1e-5,1e-6,2e-5,5e-5,3e-5]
lrl=[3e-6,5e-6,1e-5,1e-6,2e-6]

now = datetime.now()
print(f"start load data：{now}")
# Load dataset  



do1=True
edo1=True

run1=round(float(sys.argv[1]))
run2=round(float(sys.argv[2]))
cudaid=round(float(sys.argv[3]))

torch.cuda.set_device(cudaid)
torch.cuda.empty_cache()
torch.set_num_threads(5)

ep1=round(float(sys.argv[4]))
ep2=round(float(sys.argv[5]))
ep2=min(ep2,60)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def store_batches_on_gpu(dataloader, device):
    """
    将 dataloader 中的所有批次数据加载到 GPU 并存储在列表中
    :param dataloader: 数据加载器
    :param device: 设备（如 'cuda'）
    :return: 存储在 GPU 上的批次数据列表
    """
    gpu_batches = []
    for batch in dataloader:
        gpu_batches.append(batch.to(device))  # 将每个 batch 加载到 GPU
    return gpu_batches
    
def nnnbatch_predict(device,gpu_batches, model):
    #use_cuda = torch.cuda.is_available()  
    #device = torch.device("cuda" if use_cuda else "cpu")
    results = []
    with torch.no_grad():
        for batch in gpu_batches:
            output = model(batch)
            results.append(output.clone().detach())
            output=None
    # 在循环结束后一次性传输到 CPU
    results = torch.cat(results, dim=0).cpu().numpy()
    return results



mymodel = ViT().to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

for runii in range(run1,run2):

    # 获取 GPU 使用信息
    gpus = GPUtil.getGPUs()
    
    for gpu in gpus[cudaid:cudaid+1]:
        print("------------------------------------")
        print(f"GPU ID: {gpu.id}")
        print(f"名称: {gpu.name}")
        print(f"显存使用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
        print(f"GPU 使用率: {gpu.load * 100:.2f}%")
        print(f"温度: {gpu.temperature}°C")

    

    path1="//home/quhan/ML_try/dataset/newset/n224set10/tncenters1frandomrot_jpgCSSTDG_"+str(runii)+"/"
    path2="/home/quhan/ML_try/dataset/newset/n224set10/tnelsecenters1frandomrot_jpgCSSTDG_"+str(runii)+"/"
    mymodelpaths1="newresult/n224_set10/tndgmodel_centers1_"+str(runii)+""
    mymodelpaths2="newresult/n224_set10/tndgmodel_elsecenters1_"+str(runii)+""

    print("============================================path",run1,run2)
    print(path1)
    print(path2)
    print(mymodelpaths1)
    print(mymodelpaths2)

    
    if do1:
        dataset0 = load_dataset(path1)  
        print(dataset0)  
        dgset=dataset0['test'][0:len(dataset0['test'])]['image']
        
        now = datetime.now()
        print(f"finish load data：{now}")
        localtime = time.asctime( time.localtime(time.time()) )
        times=localtime.split()[3]        
        print(localtime)
        now = datetime.now()
        print(f"time：{now}")
        
    if edo1:
        datasete0 = load_dataset(path2)  
        print(datasete0)  
        edgset=datasete0['test'][0:len(datasete0['test'])]['image']
        now = datetime.now()
        print(f"finish load data：{now}")
        localtime = time.asctime( time.localtime(time.time()) )
        times=localtime.split()[3]        
        print(localtime)


    not_finish=False
    for tryi in range(100):
        BATCH_SIZE=40
        #lri=-1
        lri=0
        print("\n\n=====================================================================")
        print("tryi",tryi)
        print("\n\n=====================================================================")
        # 获取 GPU 使用信息
        gpus = GPUtil.getGPUs()
        now = datetime.now()
        print(f"time：{now}")
        for gpu in gpus[cudaid:cudaid+1]:
            print("tryi------------------------------------",tryi)
            print(f"GPU ID: {gpu.id}")
            print(f"名称: {gpu.name}")
            print(f"显存使用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
            print(f"GPU 使用率: {gpu.load * 100:.2f}%")
            print(f"温度: {gpu.temperature}°C")
        
        
        for LEARNING_RATE  in lrl[0:1]:
            lri+=1
            if do1:
                
                pathnn="/home/quhan/ML_try/result/"+str(mymodelpaths1)+"_c1flr_"+str(lri)+"/"
                pathrr=pathnn
                print("---------pathrr:",pathrr)
                print("ep:",ep1,ep2)
                gpus = GPUtil.getGPUs()
        
                for gpu in gpus[cudaid:cudaid+1]:
                    print("do ------------------------------------")
                    print(f"GPU ID: {gpu.id}")
                    print(f"名称: {gpu.name}")
                    print(f"显存使用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
                    print(f"GPU 使用率: {gpu.load * 100:.2f}%")
                    print(f"温度: {gpu.temperature}°C")

                not_finish=False
                if not os.path.exists(pathrr+"/result_dg"+str(ep2-1)+".npy"):
                    not_finish=True
                if not_finish:
                    torch.cuda.set_device(cudaid)
                    torch.cuda.empty_cache()
                    torch.set_num_threads(5)
                    images=dgset
                    images = [transform(img) for img in images]
                    dataloader = DataLoader(images, batch_size=64)
                    now = datetime.now()
                    print(f"time：{now}")
                    gpu_batches=store_batches_on_gpu(dataloader, device)
                    
                    for epi in range(ep1,ep2):
                        print("====================------------------------",epi)
                        result_nlens=[]
                        result_lens=[]
                        save_path=pathrr+"/model_test_ephoch"+str(epi)+".model"
                        
                        # Load nodel, loss function, and optimizer
                        
                        # 指定模型文件的路径
                        load_path =save_path
                        print(save_path)
                        if os.path.exists(load_path):
                            
            
                            ##============================================================================================================= s1
                        
                            do=False
                            
                            if not os.path.exists(pathrr+"/result_dg"+str(epi)+".npy"):
                                do=True
                            else:
                                result=np.load(pathrr+"/result_dg"+str(epi)+".npy")
                                if len(result.shape)==1:
                                    do=True
                            
                            if do:
                                # 加载模型的参数（权重和偏置等）
                                mymodel.load_state_dict(torch.load(load_path))
                    
                                # 将模型设置为评估模式
                                mymodel.eval()
                            
                                now = datetime.now()
                                print(f"start predict ndg：{now}")
                                list_lens=np.zeros((0,2))
                                
                                #images=dgset
                                #list_lens=batch_predict(device,images, mymodel, batch_size=64)
                                list_lens=nnnbatch_predict(device,gpu_batches, mymodel)
                                    
                                    
                                torch.cuda.empty_cache()
                                #del output
                                list_lens=np.array(list_lens)
                                #print("-------------------------------------------------------------------------------")
            
                                np.save(pathrr+"/result_dg"+str(epi),list_lens)
                                now = datetime.now()
                                
                                print(f"finish predict ndg：{now}")
                                
                gpu_batches=None
                torch.cuda.empty_cache()
        
            if edo1:
                pathnn=pathnn="/home/quhan/ML_try/result/"+str(mymodelpaths2)+"_c1flr_"+str(lri)+"/"
                pathrr=pathnn
                gpus = GPUtil.getGPUs()
        
                for gpu in gpus[cudaid:cudaid+1]:
                    print("edo ------------------------------------")
                    print(f"GPU ID: {gpu.id}")
                    print(f"名称: {gpu.name}")
                    print(f"显存使用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
                    print(f"GPU 使用率: {gpu.load * 100:.2f}%")
                    print(f"温度: {gpu.temperature}°C")

    
                not_finish=False
                if not os.path.exists(pathrr+"/result_dg"+str(ep2-1)+".npy"):
                    not_finish=True
                if not_finish:
                    torch.cuda.set_device(cudaid)
                    torch.cuda.empty_cache()
                    torch.set_num_threads(5)
                    images=edgset
                    images = [transform(img) for img in images]
                    dataloader = DataLoader(images, batch_size=64)
                    now = datetime.now()
                    print(f"time：{now}")
                    egpu_batches=store_batches_on_gpu(dataloader, device)
                
                    for epi in range(ep1,ep2):
                        print("====================------------------------",epi)
                        result_nlens=[]
                        result_lens=[]
                        save_path=pathrr+"/model_test_ephoch"+str(epi)+".model"
                        #use_cuda = torch.cuda.is_available()
                        #device = torch.device("cuda" if use_cuda else "cpu")
                        # Load nodel, loss function, and optimizer
                        #mymodel = ViT().to(device)
                        # 指定模型文件的路径
                        load_path =save_path
                        if os.path.exists(load_path):
            
                            ##============================================================================================================= s1
                        
                            do=False
                            if not os.path.exists(pathrr+"/result_dg"+str(epi)+".npy"):
                                do=True
                            else:
                                result=np.load(pathrr+"/result_dg"+str(epi)+".npy")
                                if len(result.shape)==1:
                                    do=True
                            if do:
                                # 加载模型的参数（权重和偏置等）
                                mymodel.load_state_dict(torch.load(load_path))
                    
                                # 将模型设置为评估模式
                                mymodel.eval()
                            
                                now = datetime.now()
                                print(f"start predict ndg：{now}")
                                list_lens=np.zeros((0,2))
                                
                                #images=edgset
                                #list_lens=batch_predict(device,images, mymodel, batch_size=64)
                                list_lens=nnnbatch_predict(device,egpu_batches, mymodel)
                                    
                                    
                                torch.cuda.empty_cache()
                                #del output
                                list_lens=np.array(list_lens)
                                #print("-------------------------------------------------------------------------------")
            
                                np.save(pathrr+"/result_dg"+str(epi),list_lens)
                                now = datetime.now()
                                
                                print(f"finish predict ndg：{now}")
                egpu_batches=None
                torch.cuda.empty_cache()
            
        not_finish=False
        if do1:
            pathnn=pathnn="/home/quhan/ML_try/result/"+str(mymodelpaths1)+"_c1flr_"+str(len(lrl)-1)+"/"
            pathrr=pathnn
            if not os.path.exists(pathrr+"/result_dg"+str(ep2-1)+".npy"):
                not_finish=True

        if edo1:
            pathnn=pathnn="/home/quhan/ML_try/result/"+str(mymodelpaths2)+"_c1flr_"+str(len(lrl)-1)+"/"
            pathrr=pathnn
            if not os.path.exists(pathrr+"/result_dg"+str(ep2-1)+".npy"):
                not_finish=True
                
        if not_finish:        
            time.sleep(300)
