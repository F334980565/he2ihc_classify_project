import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor

SLICE_DICT = {'C113327':0, 'A15520':1, 'A007418':0, 'C152536':1, 'A009798':0, 'A154421':0, 'A16746':1, 'C105560':0, 'A17244':1, 'C152221':1, 'A16886':1, 'A013564':0, 'A14946':0, 'A15331':1, 'A14053':1, 'C104494':0, 'A13923':1, 'A012607':1, 'C152280':1}

class LabeledDataset(Dataset):
    def __init__(self, positive_path, negative_path, bg_path, normalize=True):
        self.transform = self.get_transform(normalize)
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.bg_path = bg_path
        
        self.positive_list = os.listdir(positive_path)
        self.negative_list = os.listdir(negative_path)
        self.bg_list = os.listdir(bg_path)

    def __len__(self):
        
        return len(self.positive_list) + len(self.negative_list) + len(self.bg_list)

    def __getitem__(self, idx):
        
        if idx < len(self.positive_list):
            img_path = os.path.join(self.positive_path, self.positive_list[idx])
            label = 0
        elif idx < len(self.positive_list) + len(self.negative_list):
            img_path = os.path.join(self.negative_path, self.negative_list[idx - len(self.positive_list)])
            label = 1
        else:
            img_path = os.path.join(self.bg_path, self.bg_list[idx - len(self.positive_list) - len(self.negative_list)])
            label = 2
            
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
    
    def get_transform(self, normalize):
        class Invert(object):
            def __call__(self, image):
                return 1 - image
        
        if normalize:
            transform = Compose([
                Resize((512, 512)),
                ToTensor(),
                Invert(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = Compose([
                Resize((512, 512)),
                ToTensor(),
                Invert(),
            ])
        
        return transform
    
class CompleteDataset(Dataset):
    def __init__(self, path, is_train=True):
        self.transform = self.get_transform(is_train)
        self.path = path
        
        self.slice_list = sorted(os.listdir('/root/projects/wu/Dataset/P63_ruxian_1024'))
        self.slice_list.remove('ping')
        self.slice_dict = {}
        self.index_ranges = {}
        current_index = 0
                
        for slice_name in self.slice_list:
            slice_file_list = sorted(os.listdir(os.path.join(path, slice_name, 'he')))
            slice_he_paths = [os.path.join(path, slice_name, 'he', filename) for filename in slice_file_list]
            slice_ihc_paths = [os.path.join(path, slice_name, 'ihc', filename) for filename in slice_file_list]
            self.slice_dict[slice_name] = [slice_he_paths, slice_ihc_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
            print(slice_len)
        
    def __len__(self):
        
        n = 0
        for slice in self.slice_list:
            n += len(self.slice_dict[slice][0])
            
        return n

    def __getitem__(self, idx):
        
        for slice_name, (start, end) in self.index_ranges.items(): #似乎由潜在的bug，在某个切片里Patch特别少的时候发生，但我不知道怎么debug md
            if start <= idx < end:
                relative_index = idx - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                ihc_path = self.slice_dict[slice_name][1][relative_index]
                cur_slice = slice_name
                break
        if 'he_path' not in locals():
            print(f'index:{idx}')
            print(self.index_ranges.items())
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)
        
        he_img = 1 - he_img
        ihc_img = 1 - ihc_img
        
        ihc_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ihc_img)
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        
        return {'he': he_img, 'ihc': ihc_img, 'he_path': he_path, 'ihc_path': ihc_path, 'slice':cur_slice}
    
    def get_transform(self, train=True):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)

class IHC_Dataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.transform = self.get_transform(is_train)
        df = pd.read_csv(csv_path)

        # 提取 predict 和 img_path
        self.predicts = df['predict'].tolist()
        self.img_paths = df['img_path'].tolist()
        
    def __len__(self):
        return len(self.predicts)

    def __getitem__(self, idx):
        
        predict, img_path = self.predicts[idx], self.img_paths[idx]
        slice = img_path.split('/')[-3]
        ihc_path = img_path
        he_path = img_path.replace('ihc', 'he') #应该是ok
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)
        
        he_img = 1 - he_img #从这里开始吧
        ihc_img = 1 - ihc_img
        
        ihc_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ihc_img)
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        
        return {'he': he_img, 'ihc': ihc_img, 'he_path': he_path, 'ihc_path': ihc_path, 'predict':predict, 'slice':slice}
    
    def get_transform(self, train=True):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)

class HE_Dataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.transform = self.get_transform(is_train)
        df = pd.read_csv(csv_path)

        # 提取 predict 和 img_path
        self.all_predicts = df['predict'].tolist()
        self.all_img_paths = df['img_path'].tolist()
        
        n = len(self.all_predicts)
        cutoff = int(n*0.7)
        if is_train:
            self.predicts = self.all_predicts[:cutoff]
            self.img_paths = self.all_img_paths[:cutoff]
        else:
            self.predicts = self.all_predicts[cutoff:]
            self.img_paths = self.all_img_paths[cutoff:]
        
    def __len__(self):
        return len(self.predicts)

    def __getitem__(self, idx):
        
        predict, img_path = self.predicts[idx], self.img_paths[idx]
        slice = img_path.split('/')[-3]
        ihc_path = img_path
        he_path = img_path.replace('ihc', 'he') #应该是ok
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)
        
        he_img = 1 - he_img 
        ihc_img = 1 - ihc_img
        
        ihc_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ihc_img)
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        
        return {'he': he_img, 'ihc': ihc_img, 'he_path': he_path, 'ihc_path': ihc_path, 'predict':predict, 'slice':slice}
    
    def get_transform(self, train=True):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)
    
class HE_all_Dataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.transform = self.get_transform(is_train)
        df = pd.read_csv(csv_path)

        # 提取 predict 和 img_path
        self.predicts = df['predict'].tolist()
        self.img_paths = df['img_path'].tolist()
        
    def __len__(self):
        return len(self.predicts)

    def __getitem__(self, idx):
        
        predict, img_path = self.predicts[idx], self.img_paths[idx]
        slice = img_path.split('/')[-3]
        ihc_path = img_path
        he_path = img_path.replace('ihc', 'he') #应该是ok
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)
        
        he_img = 1 - he_img 
        ihc_img = 1 - ihc_img
        
        ihc_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ihc_img)
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        
        return {'he': he_img, 'ihc': ihc_img, 'he_path': he_path, 'ihc_path': ihc_path, 'predict':predict, 'slice':slice}
    
    def get_transform(self, train=True):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)
    
class HEtestDataset(Dataset): 
    def __init__(self, csv_path, is_train=True):
        self.transform = self.get_transform(is_train)
        df = pd.read_csv(csv_path)

        # 提取 predict 和 img_path
        self.predicts = df['predict'].tolist()
        self.img_paths = df['img_path'].tolist()
        self.slice_list = list(set([img_path.split('/')[-3] for img_path in self.img_paths]))
        self.slice_inform = {}
        
        for predict, img_path in zip(self.predicts, self.img_paths):
            slice_name = img_path.split('/')[-3] 
            if slice_name not in self.slice_inform:
                self.slice_inform[slice_name] = {}
            if predict not in self.slice_inform[slice_name]:
                self.slice_inform[slice_name][predict] = 0
            self.slice_inform[slice_name][predict] += 1
            
        print('包含切片：', self.slice_list)
        print('切片信息：', self.slice_inform)
        
    def __len__(self):
        return len(self.predicts)

    def __getitem__(self, idx):
        
        predict, img_path = self.predicts[idx], self.img_paths[idx]
        slice = img_path.split('/')[-3]
         
        ihc_path = img_path
        he_path = img_path.replace('ihc', 'he') #应该是ok
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)
        
        he_img = 1 - he_img
        ihc_img = 1 - ihc_img
        
        ihc_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ihc_img)
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        
        return {'he': he_img, 'ihc': ihc_img, 'he_path': he_path, 'ihc_path': ihc_path, 'predict':predict, 'slice':slice}
    
    def get_transform(self, train=True):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)
    
dataset = HEtestDataset(csv_path = '/root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv')