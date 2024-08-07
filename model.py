import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model = models.resnet34(pretrained=True)

# 修改最后一个全连接层以适应 3 类分类
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

class IHC_classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(IHC_classifier, self).__init__()
        
        # 加载预训练的 ResNet-34 模型
        self.classifier = models.resnet34(pretrained=True)
        
        # 修改最后一个全连接层以适应 num_classes 类分类
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Sequential(
        nn.Linear(num_features, 3),
        nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class HE_resnet34(nn.Module):
    def __init__(self, num_classes=3):
        super(HE_resnet34, self).__init__()
        
        # 加载预训练的 ResNet-34 模型
        self.classifier = models.resnet34(pretrained=True)
        
        # 修改最后一个全连接层以适应 num_classes 类分类
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Sequential(
        nn.Linear(num_features, 3),
        nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class HE_resnet50(nn.Module):
    def __init__(self, num_classes=3):
        super(HE_resnet50, self).__init__()
        
        # 加载预训练的 ResNet-50 模型
        self.classifier = models.resnet50(pretrained=True)
        
        # 修改最后一个全连接层以适应 num_classes 类分类
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Sequential(
        nn.Linear(num_features, 3),
        nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)
    
class pool256_ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(pool256_ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True) 
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4*num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 3, 4)
                
        return x
    
class pool512_ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(pool512_ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 3)
        
    def forward(self, x):
        x = self.resnet(x)
                
        return x