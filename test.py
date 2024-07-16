import os
import torch
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import LabeledDataset, CompleteDataset,FrozenHEDataset
from model import HE_resnet50, IHC_classifier
from tqdm import tqdm
import data
import csv

def test_labled(model_path, img_save_path):
    os.makedirs(img_save_path, exist_ok=True)
    
    def save_img(img_tensor, save_path):
        img = 1 - img_tensor
        img = transforms.ToPILImage()(img)
        img.save(save_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_dataset = LabeledDataset(positive_path = '/root/projects/wu/classify_project/positive_refer',
                            negative_path = '/root/projects/wu/classify_project/negative_refer',
                            bg_path = '/root/projects/wu/classify_project/non_refer')

    test_loader = DataLoader(labeled_dataset, batch_size=1, shuffle=True) 

    # 创建 ResNet-34 模型
    model = IHC_classifier()

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # 评估模型
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, img_path in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            prob, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if not predicted == labels:
                save_img(images[0], os.path.join(img_save_path, f'{predicted.data.item()}_{prob.data.item()}_{os.path.basename(img_path[0])}'))
                print(f'置信度：{prob.data}')

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    

def test_IHC(model, state_dict_path, dataloader, save_path = None):
    if not save_path == None:
        os.makedirs(os.path.join(save_path, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    
    def save_img(img_path, save_path):
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img.save(save_path)
        
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # 评估模型
    model.eval()
    iter = 0
    total = 0
    high_probs = 0
    informs = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, img_path, slice = data['ihc'], data['ihc_path'], data['slice']
            images = images.to(device)
            outputs = model(images)
            prob, predicted = torch.max(outputs.data, 1)
            #print(f'置信度：{prob}, 预测值：{predicted}, 路径：{img_path}')
            total += images.size()[0]
            informs.append((predicted.data.item(), prob.data.item(), img_path[0]))
            if total % 5 == 0:
                save_img(img_path[0], os.path.join(save_path, 'img', f'{predicted.data.item()}_{prob.data.item()}_{os.path.basename(img_path[0])}'))
            
            if prob.data.item() > 0.9:
                high_probs += images.size()[0]
            
    high_prob_rate = high_probs / total
    print(f'high prob rate on complete dataset: {high_prob_rate:.2f}%')
    
    with open(os.path.join(save_path, 'csv', 'probs.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predict', 'prob', 'img_path'])
        for pred, prob, path in informs:
            writer.writerow([pred, prob, path])


def test_HE(model, state_dict_path, dataloader, save_path = None, device='cuda:0'):
    if not save_path == None:
        os.makedirs(os.path.join(save_path, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    
    def save_img(he_path, ihc_path, save_path):
        he_img = Image.open(he_path)
        ihc_img = Image.open(ihc_path)
        
        # Get the dimensions of the images
        he_width, he_height = he_img.size
        ihc_width, ihc_height = ihc_img.size

        # Create a new blank image with the combined width and maximum height
        total_width = he_width + ihc_width
        max_height = max(he_height, ihc_height)
        new_img = Image.new('RGB', (total_width, max_height))

        # Paste the images into the new image
        new_img.paste(he_img, (0, 0))
        new_img.paste(ihc_img, (he_width, 0))

        # Save the concatenated image
        new_img.save(save_path)

    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # 评估模型
    model.eval()
    iter = 0
    total = 0
    high_probs = 0
    informs = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            he, ihc, he_path, ihc_path, slice = data['he'].to(device), data['ihc'].to(device), data['he_path'], data['ihc_path'], data['slice']
            outputs = model(he)
            prob, predicted = torch.max(outputs.data, 1)
            #print(f'置信度：{prob}, 预测值：{predicted}, 路径：{img_path}')
            total += he.size()[0]
            informs.append((predicted.data.item(), prob.data.item(), ihc_path[0]))
            if total % 5 == 0:
                save_img(he_path[0], ihc_path[0], os.path.join(save_path, 'img', f'{predicted.data.item()}_{prob.data.item()}_{os.path.basename(ihc_path[0])}'))
            
            if prob.data.item() > 0.9:
                high_probs += he.size()[0]
            
    high_prob_rate = high_probs / total
    print(f'high prob rate on complete dataset: {high_prob_rate:.2f}%')
    
    csv_file_path = os.path.join(save_path, 'csv', 'probs.csv')
    header = ['predict', 'prob', 'img_path']
    if os.path.exists(os.path.join(save_path, 'csv', 'probs.csv')):
        # Read the existing file
        df_existing = pd.read_csv(csv_file_path)
        # Ensure the header matches
        if not all(df_existing.columns == header):
            raise ValueError("CSV file exists but the header does not match the expected header.")
        # Convert the informs to a DataFrame
        df_new = pd.DataFrame(informs, columns=header)
        # Append new rows to the existing DataFrame
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Convert the informs to a DataFrame
        df_combined = pd.DataFrame(informs, columns=header)
        
        # Save the DataFrame back to CSV
        df_combined.to_csv(csv_file_path, index=False)
"""    
model = HE_resnet50()

test_slice_list = ['B008490_frozen', 'B008330_frozen', 'B008243_frozen', 'B008012_frozen', 'B007928_frozen']
test_dataset = CompleteDataset(src_path = '/root/projects/wu/Dataset/test_P63_FROZEN', slice_list = test_slice_list, predict_path='/root/projects/wu/classify_project/probs_save/IHC_probs_2_FROZEN/csv/probs.csv')


test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


test_HE(model, 
        state_dict_path = '/root/projects/wu/classify_project/checkpoints/HE_resnet50_all/HEresnet50_20epoch.pth', 
        dataloader = test_loader,
        save_path = '/root/projects/wu/classify_project/probs_save/HE_probs_2_FROZEN_test')
"""


def test_frozenHE(model, state_dict_path, dataloader, save_path = None):
    if not save_path == None:
        os.makedirs(os.path.join(save_path, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    
    def save_img(img_tensor, save_path):
        scale = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        bias = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        img = img_tensor * scale + bias
        img = 1 - img_tensor
        img = transforms.ToPILImage()(img)
        img.save(save_path)
        
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # 评估模型
    model.eval()
    iter = 0
    total = 0
    high_probs = 0
    informs = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, img_path, slice = data['he'], data['he_path'], data['slice']
            images = images.to(device)
            outputs = model(images)
            prob, predicted = torch.max(outputs.data, 1)
            #print(f'置信度：{prob}, 预测值：{predicted}, 路径：{img_path}')
            total += images.size()[0]
            informs.append((predicted.data.item(), prob.data.item(), img_path[0]))
            if total % 5 == 0:
                save_img(images[0].cpu(), os.path.join(save_path, 'img', f'{predicted.data.item()}_{prob.data.item()}_{os.path.basename(img_path[0])}'))
            
            if prob.data.item() > 0.9:
                high_probs += images.size()[0]
            
    high_prob_rate = high_probs / total
    print(f'high prob rate on complete dataset: {high_prob_rate:.2f}%')
    
    with open(os.path.join(save_path, 'csv', 'probs.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predict', 'prob', 'img_path'])
        for pred, prob, path in informs:
            writer.writerow([pred, prob, path])

model = HE_resnet50()
state_dict_path = '/root/projects/wu/classify_project/checkpoints/HE_resnet50_slicedivide_2/HEresnet50_20epoch.pth'
dataset = data.CompleteDataset(src_path='/root/projects/wu/Dataset/P63_ruxian_1024',
                               slice_list=['A009798', 'A14946', 'C105560', 'C152536', 'C152221', 'A18480', 'C133447', 'C136881'],
                               is_train=False
                               )

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

test_HE(model, state_dict_path = state_dict_path, dataloader = dataloader, save_path = '/root/projects/wu/classify_project/probs_save/HE_probs_slicedivide2')



