import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import LabeledDataset, CompleteDataset, result_Dataset
from model import IHC_classifier
from tqdm import tqdm

def find_file_by_name(folder_path, file_name):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == file_name:
                return os.path.join(root, file)
    return None

def create_file_list_csv(path1, path2, csv_file): 
    
    file_list = []
    for root, dirs, files in os.walk(path1):
        for file in files:
            file_list.append(file.replace('_real_B', ''))
    
    file_path_list = []
    
    for file in file_list:
        file_path = find_file_by_name(path2, file)
        if file_path is None:
            print(file)
        else:
            file_path_list.append(file_path)

    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File Name'])
        writer.writerows([[file_path] for file_path in file_path_list])
        
def replace_img_path_in_csv(file_path, old_str, new_str):
    """
    Replace a substring in the 'img_path' column of a CSV file and overwrite the original file.

    Args:
    file_path (str): Path to the CSV file.
    old_str (str): The substring to be replaced.
    new_str (str): The new substring to replace the old one.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Replace the old substring with the new substring in the 'img_path' column
    df['img_path'] = df['img_path'].str.replace(old_str, new_str)

    # Save the modified DataFrame back to the original CSV file
    df.to_csv(file_path, index=False)
        
def create_heatmap(csv_path, save_path, map_name):
    os.makedirs(save_path, exist_ok=True)
    
    data = pd.read_csv(csv_path)
    predicts = data['predict']
    probs = data['prob']
    img_paths = data['img_path']
    coords = []
    for img_path in img_paths:
        file_name = img_path.split('/')[-1].replace('.png', '')
        x = int(file_name.split('_')[-2].replace('x', ''))
        y = int(file_name.split('_')[-1].replace('y', ''))
        coords.append((x, y))
    
    slice_data = {}
    # 拆分出每个slice的predicts、probs和img_paths
    
    for predict, prob, img_path, coord in zip(predicts, probs, img_paths, coords):
        slice_name = img_path.split('/')[-3]  # 获取slice名称
        if slice_name not in slice_data:
            slice_data[slice_name] = {'predicts': [], 'probs': [], 'img_paths': [], 'coords': []}
        
        slice_data[slice_name]['predicts'].append(predict)
        slice_data[slice_name]['probs'].append(prob)
        slice_data[slice_name]['coords'].append(coord)
        slice_data[slice_name]['img_paths'].append(img_path)
    
    for slice_name, data in slice_data.items():
        index = [tuple(xy//1024 for xy in coord) for coord in data['coords']]
        max_x = max(coord[0] for coord in index)
        max_y = max(coord[1] for coord in index)
        heatmap_data = np.zeros((max_y + 1, max_x + 1))
        for predict, prob, idx in zip(data['predicts'], data['probs'], index):
            x, y = idx
            if predict == 0:
                heatmap_data[y][x] = 1*prob
            elif predict == 1:
                heatmap_data[y][x] = -1*prob
        #for img_path, coord in zip(data['img_paths'], data['coords']):
                
        plt.imshow(heatmap_data, cmap='bwr', interpolation='nearest')
        plt.savefig(os.path.join(save_path, f'{slice_name}_{map_name}.png'))

def result_analyze(result_path, save_path = None, device = 'cuda:0'):
    
    def create_heatmap_result(csv_path, save_path, map_name):
        os.makedirs(save_path, exist_ok=True)
        
        data = pd.read_csv(csv_path)
        predicts = data['predict']
        probs = data['prob']
        img_paths = data['img_path']
        
        coords = []
        for img_path in img_paths:
            file_name = img_path.split('/')[-1].replace('.png', '')
            x = int(file_name.split('_')[-4].replace('x', ''))
            y = int(file_name.split('_')[-3].replace('y', ''))
            coords.append((x, y))
        
        slice_data = {}
        # 拆分出每个slice的predicts、probs和img_paths
        
        for predict, prob, img_path, coord in zip(predicts, probs, img_paths, coords):
            slice_name = img_path.split('/')[-1].split('_')[0]  # 获取slice名称
            if slice_name not in slice_data:
                slice_data[slice_name] = {'predicts': [], 'probs': [], 'img_paths': [], 'coords': []}
            
            slice_data[slice_name]['predicts'].append(predict)
            slice_data[slice_name]['probs'].append(prob)
            slice_data[slice_name]['coords'].append(coord)
            slice_data[slice_name]['img_paths'].append(img_path)
        
        for slice_name, data in slice_data.items():
            index = [tuple(xy//1024 for xy in coord) for coord in data['coords']]
            max_x = max(coord[0] for coord in index)
            max_y = max(coord[1] for coord in index)
            heatmap_data = np.zeros((max_y + 1, max_x + 1))
            for predict, prob, idx in zip(data['predicts'], data['probs'], index):
                x, y = idx
                if predict == 0:
                    heatmap_data[y][x] = 1*prob
                elif predict == 1:
                    heatmap_data[y][x] = -1*prob
            #for img_path, coord in zip(data['img_paths'], data['coords']):
                    
            plt.imshow(heatmap_data, cmap='bwr', interpolation='nearest')
            plt.savefig(os.path.join(save_path, f'{slice_name}_{map_name}.png'))
    
    if not save_path == None:
        os.makedirs(save_path, exist_ok=True)
    
    def save_img(img_tensor, save_path):
        scale = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        bias = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        img = img_tensor * scale + bias
        img = 1 - img_tensor
        img = transforms.ToPILImage()(img)
        img.save(save_path)
    
    model = IHC_classifier()
    state_dict = torch.load('/root/projects/wu/classify_project/checkpoints/IHCclassifier/IHCclassifer_10epoch.pth')
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = result_Dataset(result_path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 评估模型
    model.eval()
    iter = 0
    total = 0
    high_probs = 0
    real_informs = []
    fake_informs = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            he_img, real_img, fake_img, real_path, fake_path, slice = data['he'], data['real'], data['fake'], data['real_path'], data['fake_path'], data['slice']
            
            real_img = real_img.to(device)
            fake_img = fake_img.to(device)
            
            real_outputs = model(real_img)
            fake_outputs = model(fake_img)
            real_prob, real_predicted = torch.max(real_outputs.data, 1)
            fake_prob, fake_predicted = torch.max(fake_outputs.data, 1)
            
            #print(f'置信度：{prob}, 预测值：{predicted}, 路径：{img_path}')
            total += real_img.size()[0]
            real_informs.append((real_predicted.data.item(), real_prob.data.item(), real_path[0]))
            fake_informs.append((fake_predicted.data.item(), fake_prob.data.item(), fake_path[0]))
            
            """
            理论上不需要存图，假定IHC分类器已经完善
            if total % 5 == 0:
                save_img(images[0].cpu(), os.path.join(save_path, 'img', f'{predicted.data.item()}_{prob.data.item()}_{os.path.basename(img_path[0])}'))
            """
            
            if fake_prob.data.item() > 0.95:
                high_probs += fake_img.size()[0]
            
    high_prob_rate = high_probs / total * 100
    print(f'high prob rate on complete dataset: {high_prob_rate:.2f}%')
    
    with open(os.path.join(save_path, 'real_probs.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predict', 'prob', 'img_path'])
        for pred, prob, path in real_informs:
            writer.writerow([pred, prob, path])
            
    with open(os.path.join(save_path, 'fake_probs.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predict', 'prob', 'img_path'])
        for pred, prob, path in fake_informs:
            writer.writerow([pred, prob, path])
    
    create_heatmap_result(os.path.join(save_path, f'real_probs.csv'), os.path.join(save_path, 'heatmap'), map_name='real')
    create_heatmap_result(os.path.join(save_path, f'fake_probs.csv'), os.path.join(save_path, 'heatmap'), map_name='fake')


"""
result_analyze(result_path = '/root/projects/wu/he2ihc/results/betatest_P63fenlei/test_80/images',
               save_path = '/root/projects/wu/he2ihc/results/betatest_P63fenlei/test_80/result_analyze',
               device = 'cuda:0')
"""

"""
create_heatmap(csv_path = '/root/projects/wu/classify_project/probs_save/HE_probs_slicedivide2/csv/probs.csv',
               save_path = '/root/projects/wu/classify_project/probs_save/HE_probs_slicedivide2/heatmap',
               map_name = 'heatmap')
"""

file_path = '/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_all/csv/probs.csv'
old_str = '/root/projects/wu/Dataset/P63_ruxian_1024'
new_str = '/home/f611/Projects/data/Dataset_171/P63_ruxian_1024'
replace_img_path_in_csv(file_path, old_str, new_str)
