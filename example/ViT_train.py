import transformers
import time
import math
import random
import os
import sys
from datetime import datetime

import numpy as np  
import torch  
import cv2  
import torch.nn as nn  
from transformers import ViTModel, ViTConfig  
from torchvision import transforms  
from torch.optim import Adam  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
from datasets import load_dataset  


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

#Pretrained model checkpoint  
model_checkpoint = '../vit-base-patch16-224-in21k'  

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
        model_checkpoint= '../vit-base-patch16-224-in21k'):  
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
def model_train(dataset, epochs, learning_rate, bs,pathnn):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(use_cuda,device)

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
        save_path=pathnn+"/model_test_ephoch"+str(i)+".model"
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



now = datetime.now()
print(f"start load data：{now}")

cudaid=0
cudaid1=cudaid
cudaid2=cudaid

mkdir("../result/n224_set10/")

if round(float(sys.argv[1]))==1:
    do1=True
else:
    do1=False
    
if round(float(sys.argv[2]))==1:
    edo1=True
else:
    edo1=False

## 超参数设置
lrl=[float(sys.argv[3])]
EPOCHS = 60
BATCH_SIZE = 40


pathset1="../dataset/sampleA_randomrot_jpgCSSTDG/"
pathset2="../dataset/sampleB_randomrot_jpgCSSTDG/"
model_spath1="../result/dgmodel_sampleA"
model_espath1="../result/dgmodel_sampleB"

lri=0
for LEARNING_RATE  in lrl:
    lri+=1
    if do1:
        pathnn=str(model_spath1)+"_"+str(lri)+"/"
        mkdir(pathnn)

        print("run:",model_spath1,"lr=",LEARNING_RATE)
        now = datetime.now()
        print(f"time：{now}")

        torch.cuda.set_device(cudaid)
        torch.set_num_threads(1)
        dataset = load_dataset(pathset1)  
        print(dataset)  
        now = datetime.now()
        print(f"finish load data：{now}")
        localtime = time.asctime( time.localtime(time.time()) )
        times=localtime.split()[3]        
        print(localtime)
    
        now = datetime.now()
        print("---------------------------------------------------")
        print(f"start train：{now}",lri,LEARNING_RATE)
        print("---------------------------------------------------\n")
       
        
        print("\nEPOCHS =",EPOCHS)
        print("LEARNING_RATE =",LEARNING_RATE)
        print("BATCH_SIZE =",BATCH_SIZE)
        print("\n   ")
        
        trained_model = model_train(dataset['train'], EPOCHS, LEARNING_RATE, BATCH_SIZE,pathnn)
        save_path=pathnn+"dg_model_test.model"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(trained_model.state_dict(), save_path)

    if edo1:
        pathnn=str(model_espath1)+"_"+str(lri)+"/"
        mkdir(pathnn)
        if not os.path.exists(pathnn+"dg_model_test.model"):
            print("run:",model_espath1,"lr=",LEARNING_RATE)
            now = datetime.now()
            print(f"time：{now}")
        
            
            os.system("cp "+current_file_path+" "+pathnn)
            
            torch.cuda.set_device(cudaid)
            torch.set_num_threads(1)
            dataset = load_dataset(pathset2)  
            print(dataset)  
            now = datetime.now()
            print(f"finish load data：{now}")
            localtime = time.asctime( time.localtime(time.time()) )
            times=localtime.split()[3]        
            print(localtime)
        
        
            now = datetime.now()
            print("---------------------------------------------------")
            print(f"start train：{now}",lri,LEARNING_RATE)
            print("---------------------------------------------------\n")
            ## 超参数设置
            
            print("\nEPOCHS =",EPOCHS)
            print("LEARNING_RATE =",LEARNING_RATE)
            print("BATCH_SIZE =",BATCH_SIZE)
            print("\n   ")
            
            
            trained_model = model_train(dataset['train'], EPOCHS, LEARNING_RATE, BATCH_SIZE,pathnn)
            save_path=pathnn+"dg_model_test.model"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(trained_model.state_dict(), save_path)
