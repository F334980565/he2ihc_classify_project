import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
def create_heatmap(csv_path, save_path):
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
        plt.savefig(os.path.join(save_path, f'{slice_name}_heatmap.png'))

def temp():
    df_high_probs = pd.read_csv('/root/projects/wu/classify_project/high_probs_save/csv/high_prob.csv')
    df_he = pd.read_csv('/root/projects/wu/classify_project/probs_save/HE_probs_1/csv/probs.csv')
    
    for img_path in df_high_probs['img_path'].tolist():
        df_he
 
csv_path = '/root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv'
save_path = '/root/projects/wu/classify_project/probs_save/IHC_probs_2/heatmap'
create_heatmap(csv_path, save_path)
