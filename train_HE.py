#%%
import os
import torch
from model import IHC_classifier, HE_resnet34, HE_resnet50, pool_ResNet50
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from data import LabeledDataset, CompleteDataset, HE_Dataset

def train(model, train_loader, num_epochs, save_path, model_name, checkpoint=None):
    os.makedirs(save_path, exist_ok=True)
    num_epochs = num_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class_weights = torch.tensor([3, 2, 0.5], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    
    if not checkpoint is None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        n = 0
        for data in tqdm(train_loader):
            images = data['he'].to(device)
            labels = data['predict'].to(device)
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
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
                print(f'Current iter:{n}, Current accuracy: {cur_accuracy:.2f}%')
            n += 1
            
        scheduler.step()
        accuracy = 100 * correct / total
        
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss.item():.4f}, Accuracy:{accuracy:.2f}%')

        torch.save(model.state_dict(), os.path.join(save_path ,f'{model_name}_{epoch+1}epoch.pth'))
        print(f'Model saved')

def train_pool(model, train_loader, num_epochs, save_path, model_name, checkpoint=None):
    os.makedirs(save_path, exist_ok=True)
    num_epochs = num_epochs
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    
    if not checkpoint is None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        n = 0
        for data in tqdm(train_loader):
            images = data['he'].to(device)
            labels = data['predict_tensor'].to(device)
            # 前向传播
            outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            #_, predictions = torch.max(outputs, dim=2)  # 获取预测的类别索引
            
            label_tensor = torch.where(labels == 2, -1, torch.where(labels == 0, 1, torch.where(labels == 1, 0, labels)))
            loss = criterion(outputs, label_tensor)
            total_loss += loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            total += labels.size(0)*4
            
            predictions = torch.where(outputs > 0.5, 1,
                     torch.where(outputs <= 0.5, 0, torch.where(outputs < -0.5, -1, outputs)))
            correct += (predictions == label_tensor).sum().item()
            cur_accuracy = 100 * correct / total
            if n % 10 == 0:
                print(f'Current iter:{n}, Current accuracy: {cur_accuracy:.2f}%')
            n += 1
            
        scheduler.step()
        accuracy = 100 * correct / total
        
        average_loss = total_loss / len(train_loader)
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss.item():.4f}, Accuracy:{accuracy:.2f}%')

        torch.save(model.state_dict(), os.path.join(save_path ,f'{model_name}_{epoch+1}epoch.pth'))
        print(f'Model saved')

def test(model, test_loader, model_state_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state_dict = torch.load(model_state_path)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            images = data['he'].to(device)
            labels = data['predict'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy
    
train_slice_list = ['C113327', 'A15520', 'A007418', 'A154421', 'A16746', 'A17244', 'A16886', 'A013564', 'A15331', 'A14053', 'C104494', 'A13923', 'A012607', 'C152280', 'A8827', 'A10032']
train_dataset = CompleteDataset(src_path = '/home/f611/Projects/data/Dataset_171/P63_ruxian_1024', slice_list = train_slice_list, predict_path='/home/f611/Projects/wu/he2ihc_classify_project/results/IHC_pool_all/labels_2.csv')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = pool_ResNet50()

train_pool(model, train_loader, num_epochs = 20, save_path = '/home/f611/Projects/wu/he2ihc_classify_project/checkpoints/pool_classifier', model_name = 'pool_classifier_unet', checkpoint=None)
#train(model, train_loader, num_epochs=20, save_path='/root/projects/wu/classify_project/checkpoints/HE_resnet50_slicedivide_2', model_name='HEresnet50', checkpoint='/root/projects/wu/classify_project/checkpoints/HE_resnet50_slicedivide/HEresnet50_10epoch.pth')
# %%
