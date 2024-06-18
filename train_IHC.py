import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from data import LabeledDataset, CompleteDataset, IHC_Dataset
from model import IHC_classifier
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
    model = models.resnet34(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 3),
        nn.Softmax(dim=1)
    )

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
    
def train_unlabled(model, dataloader, num_epoch, device, save_path):
    os.makedirs(save_path, exist_ok=True)
        
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(num_epoch):
        correct = 0
        total = 0
        total_loss = 0
        n = 0
        for data in dataloader:
            images = data['ihc'].to(device)
            labels = data['predict'].to(device)
            slice = data['slice']
            # 前向传播
            outputs = model(images)
            prob, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            total_loss += loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cur_accuracy = 100 * correct / total
            if n % 10 == 0:
                print(f'Current iter:{n}, Current accuracy: {cur_accuracy:.2f}%')sZ
            n += 1

        accuracy = 100 * correct / total
        
        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {average_loss.item():.4f}, Accuracy:{accuracy:.2f}%')

        torch.save(model.state_dict(), os.path.join(save_path, f'IHCclassifer_{epoch+1}epoch.pth'))
        print(f'Model saved')

dataset = IHC_Dataset(csv_path='/root/projects/wu/classify_project/high_probs_save/csv/high_prob.csv')
data_loader = DataLoader(dataset, batch_size=31, shuffle=True)
model = IHC_classifier()

train_unlabled(model, data_loader, num_epoch=10, device='cuda:0', save_path='/root/projects/wu/classify_project/checkpoints/IHCclassifier')
