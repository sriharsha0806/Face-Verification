import math
import random
import os
import cv2
import pandas as pd
from PIL import Image, ImageDraw
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
from net import SiameseNetwork
from losses import ContrastiveLoss
import torch.nn.functional as F

class Config():
    batch_size = 2
    number_of_workers = 2
    number_epochs = 1

class SiameseNetworkDataset(Dataset):
    def __init__(self,csv_file,transform = None, should_invert=True):
        """
        Args: 
            csv_file (string): Path to the csv file with labels
            transform (callable, optional): Optional transform to be 
            applied on a sample.            
        """
        self.celeb_frame = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.celeb_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        n = random.randint(0, len(self.celeb_frame)-1)
        img0_tuple = (os.path.join(self.celeb_frame.iloc[n, 0]), self.celeb_frame.iloc[n,1])
        
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                m = random.randint(0, len(self.celeb_frame)-1)
                img1_tuple = (os.path.join(self.celeb_frame.iloc[m,0]), self.celeb_frame.iloc[m,1])
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                m = random.randint(0, len(self.celeb_frame)-1)
                img1_tuple = (os.path.join(self.celeb_frame.iloc[m,0]), self.celeb_frame.iloc[m,1])
                if img0_tuple[1] != img1_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        
        # Grey Images exist: 11452         
        if img0.mode == "L":
            cv_image = np.array(img0)
            img0 = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            img0 = Image.fromarray(img0)
            
        # Grey Images exist: 11452         
        if img1.mode == "L":
            cv_image = np.array(img1)
            img1 = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            img1 = Image.fromarray(img1)
                
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array(
        [int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32))
        

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = [key  for (key, value) in dictOfElements.items() if value == valueToFind]
    return  listOfKeys

def save_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.savefig('plot.png')

def main(csv_file):
    data = pd.read_csv(csv_file)
    print("The number of classes {} \n The number of Images in the Dataset is {}".format(len(set(data['class'])), data.shape[0]))
    histogram = {}
    
    for name in data['class']:
        if name in histogram:
            histogram[name] += 1
        else:
            histogram[name] = 1

    class_max = getKeysByValue(histogram, max(histogram.values()))
    class_min = getKeysByValue(histogram, min(histogram.values()))
    print("classes with maximum examples {} and the number of examples {}".format(class_max, max(histogram.values())))
    print("\n")
    print("number of classes with one example {}".format(len(class_min)))


    siamese_dataset = SiameseNetworkDataset(csv_file='meta.csv',
                                       transform = transforms.Compose(
                                       [transforms.Resize((224,224)),
                                       transforms.ToTensor()]))                        
                                                                             
    validation_split = 0.8
    train_size = int(validation_split*len(siamese_dataset))
    test_size = len(siamese_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(siamese_dataset,
                                                            [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=Config.number_of_workers,
                        batch_size=Config.batch_size)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    save_plot(counter,loss_history)

    print("Training Completed")

    # Save the model
    torch.save(net, '.')
    
    print("Model is saved")

if __name__ == "__main__":
    csv_file = "meta.csv"
    main(csv_file)
