#%%
import os
import pickle
import pandas as pd
from data import CompleteDataset
from model import pool256_ResNet50, pool512_ResNet50
import torch
import kornia
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from scipy.optimize import minimize
import collections

def get_range(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")
    
    
def get_predict_bymaxpool(img_tensor, params = [1, 1, 1, 1.85, 0.5]):
    
    img_tensor = (1 - img_tensor) / 2
    a, b, c, positive_threshold, background_threshold = params
    avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(8, 8))
    max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(64, 64))
    sum = a * max_pool_512.squeeze()[0] + b * max_pool_512.squeeze()[1] + c * max_pool_512.squeeze()[2]
    if sum > positive_threshold:
        predict = 0
    elif sum <= positive_threshold and sum >= background_threshold:
        predict = 1
    else:
        predict = 2
    return predict, sum

def visualizer(data):
        img_tensor = (1 - data['ihc'][0])/2
        predict = data['predict'][0]
        ihc_path = data['ihc_path'][0]
        he_path = data['he_path'][0]
        predict_tensor = data['predict_tensor'][0]
        pool_predict, sum = get_predict_bymaxpool(data['ihc'])
        #if not (sum >= 1.75 and sum < 1.95):
            #return 0
        # 这个基本就是P63 的T了，拿max_pool_256二值化
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(8, 8))
        max_pool_16 = F.max_pool2d(avg_pool_8, kernel_size=(2, 2))
        max_pool_32 = F.max_pool2d(avg_pool_8, kernel_size=(4, 4))
        max_pool_64 = F.max_pool2d(avg_pool_8, kernel_size=(8, 8))
        max_pool_128 = F.max_pool2d(avg_pool_8, kernel_size=(16, 16))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(32, 32))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(64, 64))
        """
        avg_pool_16 = F.avg_pool2d(img_tensor, kernel_size=(16, 16))
        max_pool_32 = F.max_pool2d(avg_pool_16, kernel_size=(2, 2))
        max_pool_64 = F.max_pool2d(avg_pool_16, kernel_size=(4, 4))
        max_pool_128 = F.max_pool2d(avg_pool_16, kernel_size=(8, 8))
        max_pool_256 = F.max_pool2d(avg_pool_16, kernel_size=(16, 16))
        max_pool_512 = F.max_pool2d(avg_pool_16, kernel_size=(32, 32))
        """

        # 去掉批次维度并将张量转换为 numpy 数组
        image_tensor = img_tensor.squeeze(0).permute(1, 2, 0).numpy()

        # 从文件路径读取图像
        ihc_tensor = Image.open(ihc_path)
        he_tensor = Image.open(he_path)

        # 调整图像尺寸以匹配
        image_tensor_resized = Image.fromarray((image_tensor * 255).astype(np.uint8))
        he_tensor_resized = he_tensor.resize(image_tensor.shape[:2][::-1])
        ihc_tensor_resized = ihc_tensor.resize(image_tensor.shape[:2][::-1])

        pool_results = [
        ('he', he_tensor_resized),
        ('ihc', ihc_tensor_resized),
        ('Input', image_tensor_resized),
        ('Max Pool 16', max_pool_16.squeeze(0).permute(1, 2, 0).numpy()),
        ('Max Pool 32', max_pool_32.squeeze(0).permute(1, 2, 0).numpy()),
        ('Max Pool 64', max_pool_64.squeeze(0).permute(1, 2, 0).numpy()),
        ('Max Pool 128', max_pool_128.squeeze(0).permute(1, 2, 0).numpy()),
        ('Max Pool 256', max_pool_256.squeeze(0).permute(1, 2, 0).numpy()),
        ('Max Pool 512', max_pool_512.squeeze(0).permute(1, 2, 0).numpy()),
        ('grid_he_tensor', he_tensor),
        ]

        # 创建一个图形并设置子图
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))

        # 显示每个池化结果
        for ax, (title, img) in zip(axes.flatten(), pool_results):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)

        # 在原始张量图像上添加预测值注释
        annotation_text = f'predict: {predict}\npredict_tensor: {predict_tensor}\nsum: {sum}'
        axes[0, 0].text(10, 20, annotation_text, color='white', fontsize=12, 
                        bbox=dict(facecolor='black', alpha=0.5), ha='left', va='top')

        # 显示图形
        plt.tight_layout()
        plt.show()

def pool_test():
    dataset = CompleteDataset(src_path = '/home/f611/Projects/data/Dataset_171/CK56_ruxian_1024', 
                        slice_list = 'all',
                        is_train=False,
                        predict_path='/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_all/csv/probs.csv',
                        use_pool = True)
    
    columns = ['predict_pool', 'sum', 'predict_resnet', 'ihc_path']
    results = []
    
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
    n = 0
    for data in tqdm(dataLoader):
        visualizer(data)
        n += 1
        if n > 20:
            break
    """
        results.append({
            'predict_pool': pool_predict,
            'predict_resnet': predict,
            'sum': sum,
            'ihc_path': ihc_path,
            'he_path': he_path
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_pool_all/labels_2.csv', index=False)
    """

def find_parameter(df_path):
    # 读取 CSV 文件
    df = pd.read_csv(df_path)

    # 提取 sum 列的数值
    df['sum'] = df['sum'].apply(lambda x: float(x.split('(')[1].split(')')[0]))

    # 打印提取后的 sum 列以验证
    print(df['sum'])

    # 绘制直方图
    plt.hist(df['sum'], bins=30, edgecolor='black')

    # 添加标题和标签
    plt.title('Distribution of Sum Values')
    plt.xlabel('Sum')
    plt.ylabel('Count')

    # 显示图表
    plt.show()

def train_parameter(device='cuda:0'):
    #/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_all/csv/probs.csv'
    dataset = CompleteDataset(src_path = '/home/f611/Projects/data/Dataset_171/CK56_ruxian_1024', 
                        slice_list = 'all',
                        is_train=True,
                        predict_path=None,
                        use_pool=True)
    
    dataloader = DataLoader(dataset=dataset, batch_size=12, shuffle=True)
    model = pool512_ResNet50().to(device)

    class_weights = torch.tensor([3, 1, 0.5], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    L1_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    num_epoch = 10

    model.train()
    for epoch in range(10):
        correct = 0
        total = 0
        total_loss = 0
        n = 0
        for data in tqdm(dataloader):
            img_tensor = data['he'].to(device)
            labels = data['predict'].to(device)
            img_path = data['ihc_path']
            
            outputs = model(img_tensor)
            predict = torch.argmax(outputs, 1)
            
            loss = criterion(outputs, labels)
            total_loss += loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            cur_avg_loss = total_loss / total
            correct += (predict == labels).sum().item()
            cur_accuracy = 100 * correct / total
            if n % 10 == 0:
                #print(f'Current iter:{n}, Current avg loss:{cur_avg_loss}')
                print(f'Current iter:{n}, Current avg loss:{cur_avg_loss}, Current accuracy:{cur_accuracy}')
            n += 1
            
            #accuracy = 100 * correct / total
            
            average_loss = total_loss / len(dataloader)
            
        #print(f'Epoch [{epoch+1}/2], Loss: {average_loss.item():.4f}, Accuracy:{accuracy:.2f}%')
        print(f'Epoch [{epoch+1}/2], Loss: {average_loss.item():.4f}')
        scheduler.step()

        torch.save(model.state_dict(), os.path.join('/home/f611/Projects/wu/he2ihc_classify_project/checkpoints/HEpool_classifier' ,f'pool512_CK56_2.3_0.7_classifer_{epoch+1}epoch.pth'))
        print(f'Model saved')
        
def test():
    dataset = CompleteDataset(src_path = '/home/f611/Projects/data/Dataset_171/P63_ruxian_1024', 
                        slice_list = ['C152221', 'A15520', 'A154421', 'A13923', 'A17244', 'A16746', 'C133447', 'A10032', 'A012607', 'A15331', 'C152536', 'A16886', 'C136881', 'A14946', 'A009798', 'A8827', 'A007418', 'C152280', 'A013564', 'A18480', 'A14053'],
                        is_train=False,
                        predict_path='/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_all/csv/probs.csv')
    
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    
    correct = 0
    cur_iter = 0
    shreshold = 0.70 * 3
    a = 1
    b = 1
    c = 1
    for data in tqdm(dataloader):
        img_tensor = data['ihc']
        labels = data['predict']
        img_path = data['ihc_path']
        max_pool_512 = F.max_pool2d(img_tensor, kernel_size=(512, 512))
        
        #sum = torch.sum(max_pool_512)
        sum = a * max_pool_512.squeeze()[0] + b * max_pool_512.squeeze()[1] + c * max_pool_512.squeeze()[2]
        if sum > shreshold:
            predict = 0
        else:
            predict = 1
        
        if ((labels == 1 or labels == 2) and predict == 1) or (labels == 0 and predict == 0):
            correct += 1
        
        cur_iter += 1
        cur_accuracy = 100 * correct / cur_iter
        if cur_iter % 100 == 0:
            print(f'Current iter:{cur_iter}, Current accuracy: {cur_accuracy:.2f}%')
            
    print(f'Accuracy:{cur_accuracy:.2f}%')

#%%
if __name__ == "__main__":
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import random
    
    #df_path = '/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_pool_all/labels_2.csv'
    #find_parameter(df_path)
    #pool_test()
    train_parameter(device='cuda:1')
    


# %%
