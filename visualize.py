import cv2
import numpy as np 
import torch
import torchvision
from torch import optim 
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from PIL import Image
from net import SiameseNetwork

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(140, 26, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def check_gray(img):
    if img.mode == "L":
        cv_image = np.array(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img) 
    return img

def visualize(img1, img2,device, net):
    img1 = check_gray(img1)
    img2 = check_gray(img2)
    trans_size = transforms.Resize((224,224))
    trans_tensor = transforms.ToTensor()
    img1 = trans_tensor(trans_size(img1)).unsqueeze_(0)
    img2 = trans_tensor(trans_size(img2)).unsqueeze_(0)
    net = torch.load('siamese_temp.pt', map_location="cuda:0")
    net.eval()
    output1,output2 = net(Variable(img1).cuda(),Variable(img2).cuda())
    concatenated = torch.cat((img1,img2),0) 
    euclidean_distance = F.pairwise_distance(output1, output2)
    print(euclidean_distance.item())
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

if __name__ == '__main__':
    img1 = Image.open("test5.jpeg")
    img2 = Image.open("test6.jpeg")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SiameseNetwork().to(device)
    net.eval()
    visualize(img1, img2, device, net)
    print("Completed Visualization")
