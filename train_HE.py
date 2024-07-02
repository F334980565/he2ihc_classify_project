import os
import torch
from model import IHC_classifier, HE_resnet34, HE_resnet50
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from data import LabeledDataset, CompleteDataset, HE_Dataset, HE_all_Dataset

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
    
#train_dataset = HE_Dataset(csv_path='/root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv', is_train=True) 这两是训70%训练集，30%数据集，不独立切片，看看效果
#test_dataset = HE_Dataset(csv_path='/root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv', is_train=False)
all_dataset = HE_all_Dataset(csv_path='/root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv')
train_slice_list = ['C113327', 'A15520', 'A007418', 'A154421', 'A16746', 'A17244', 'A16886', 'A013564', 'A15331', 'A14053', 'C104494', 'A13923', 'A012607', 'C152280', 'A8827', 'A10032']
train_dataset = CompleteDataset(src_path = '/root/projects/wu/Dataset/P63_ruxian_1024', slice_list = train_slice_list, predict_path='/root/projects/wu/classify_project/probs_save/IHC_probs_3_P63120all/csv/probs.csv')

train_loader = DataLoader(train_dataset, batch_size=19, shuffle=True)

model = HE_resnet50()

train(model, train_loader, num_epochs=20, save_path='/root/projects/wu/classify_project/checkpoints/HE_resnet50_slicedivide_2', model_name='HEresnet50', checkpoint='/root/projects/wu/classify_project/checkpoints/HE_resnet50_slicedivide/HEresnet50_10epoch.pth')