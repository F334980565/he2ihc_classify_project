import os
import torch
import random
import pandas as pd
from PIL import Image
import torch.nn.functional as F
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

class CompleteDataset(Dataset): #这个是读Datset_171里的数据集的时候用的
    def __init__(self, src_path, slice_list, is_train=False, predict_path=None, use_pool=False, stain_name = 'P63'):
        self.slice_list = slice_list
        self.src_path = src_path
        self.predict_path = predict_path
        self.use_pool = use_pool
        self.is_train = is_train
        self.stain_name = stain_name
        if use_pool:
            if stain_name == 'P63_temp':
                self.params = [16, 1.2, 0.5]
            if stain_name == 'P63':
                self.params = [8, 1.9, 0.5]
            elif stain_name == 'CK56':
                self.params = [8, 2.25, 0.4]
            elif stain_name == 'P16':
                self.params = [16, 2.0, 0.7]
                
        if slice_list == 'all':
            self.slice_list = sorted([name for name in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, name))])

        self.slice_dict = {}
        self.index_ranges = {}
        current_index = 0
                
        for slice_name in self.slice_list:
            slice_file_list = sorted(os.listdir(os.path.join(self.src_path, slice_name, 'he')))
            slice_he_paths = [os.path.join(self.src_path, slice_name, 'he', filename) for filename in slice_file_list]
            slice_ihc_paths = [os.path.join(self.src_path, slice_name, 'ihc', filename) for filename in slice_file_list]
            self.slice_dict[slice_name] = [slice_he_paths, slice_ihc_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
        
        if not predict_path == None:
            df = pd.read_csv(self.predict_path)
            img_paths = df['img_path'].tolist()
            predicts = df['predict'].tolist()
            predict_dict = dict(zip(img_paths, predicts))
            for slice_name in self.slice_list:
                slice_predicts = []
                ihc_paths = self.slice_dict[slice_name][1]
                for ihc_path in ihc_paths:
                    slice_predicts.append(predict_dict.get(ihc_path))
                self.slice_dict[slice_name].append(slice_predicts)
        
    def __len__(self):
        
        n = 0
        for slice in self.slice_list:
            n += len(self.slice_dict[slice][0])

        return n

    def __getitem__(self, idx):
        
        for slice_name, (start, end) in self.index_ranges.items():
            if start <= idx < end:
                relative_index = idx - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                ihc_path = self.slice_dict[slice_name][1][relative_index]
                if not self.predict_path == None:
                    predict = self.slice_dict[slice_name][2][relative_index]
                cur_slice = slice_name
                break
        
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')

        if self.is_train:
            flip = random.choice([True, False])
            transform = self.get_transform(flip)
        else:
            transform = self.get_transform()

        he_tensor = transform(he_img)
        ihc_tensor = transform(ihc_img)
        
        if self.use_pool:
            #if self.stain_name == 'CK56':
                #predict, predict_tensor = self.get_CK56_label(ihc_tensor, self.params)
            #else:
            predict, predict_tensor, sum = self.get_label_tensor(ihc_tensor, self.params)
            return_dict = {'he': he_tensor, 'ihc': ihc_tensor, 'he_path': he_path, 'ihc_path': ihc_path, 'predict_tensor':predict_tensor, 'predict':predict, 'slice':cur_slice, 'sum': sum}
        elif not self.predict_path == None:
            return_dict = {'he': he_tensor, 'ihc': ihc_tensor, 'he_path': he_path, 'ihc_path': ihc_path, 'predict':predict, 'slice':cur_slice}
        else:
            return_dict = {'he': he_tensor, 'ihc': ihc_tensor, 'he_path': he_path, 'ihc_path': ihc_path, 'slice':cur_slice}
            
        return return_dict
    
    def get_transform(self, flip=False):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=1))

        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        return transforms.Compose(transform_list)
    
    def get_label_tensor(self, img_tensor, params = [8, 2.3, 0.7]):
        img_tensor = (1 - img_tensor) / 2
        img_size = img_tensor.shape[-1]
        k, positive_threshold, background_threshold = params
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(k, k))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // (2*k), img_size // (2*k)))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // k, img_size // k))
        #sum = max_pool_512.sum()
        sum = 0.0 * max_pool_512[0] + 0.0 * max_pool_512[1] + 3.0 * max_pool_512[2]
        if sum > positive_threshold:
            predict = 0
        elif sum <= positive_threshold and sum >= background_threshold:
            predict = 1
        else:
            predict = 2
        
        summed_tensor = torch.sum(max_pool_256, dim=0)
        predict_tensor = torch.where(summed_tensor > positive_threshold, 0,
                                    torch.where((summed_tensor <= positive_threshold) & (summed_tensor >= background_threshold), 1, 2))
        
        return predict, predict_tensor, sum

    def get_CK56_label(self, img_tensor, params = [8, 2.3, 0.7]):
        img_tensor = (1 - img_tensor) / 2
        img_size = img_tensor.shape[-1]
        k, positive_threshold, background_threshold = params
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(k, k))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // (2*k), img_size // (2*k)))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // k, img_size // k))
        sum = max_pool_512[0] + max_pool_512[1]
        if sum > positive_threshold:
            predict = 0
        elif sum <= positive_threshold and sum >= background_threshold:
            predict = 1
        else:
            predict = 2
        
        summed_tensor = torch.sum(max_pool_256, dim=0)
        predict_tensor = torch.where(summed_tensor > positive_threshold, 0,
                                    torch.where((summed_tensor <= positive_threshold) & (summed_tensor >= background_threshold), 1, 2))
        
        return predict, predict_tensor

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
    
class result_Dataset(Dataset): #这个是给he2ihc的results测试分类结果准不准用的
    def __init__(self, result_path):
        self.img_files = os.listdir(result_path)
        self.he_paths = [os.path.join(result_path, file_name) for file_name in self.img_files if 'real_A' in file_name]
        self.real_paths = [path.replace('real_A', 'real_B') for path in self.he_paths]
        self.fake_paths = [path.replace('real_A', 'fake_B') for path in self.he_paths]
        self.transform = self.get_transform()
        
    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx):
        
        he_path = self.he_paths[idx]
        real_path = self.real_paths[idx]
        fake_path = self.fake_paths[idx]
        
        slice = he_path.split('/')[-1].split('_')[0]
        
        he_img = Image.open(he_path).convert('RGB')
        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')

        he_img = self.transform(he_img)
        real_img = self.transform(real_img)
        fake_img = self.transform(fake_img)
        
        he_img = 1 - he_img #从这里开始吧
        real_img = 1 - real_img
        fake_img = 1 - fake_img
        
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        real_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_img)
        fake_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fake_img)
        
        return {'he': he_img, 'real': real_img, 'fake': fake_img, 'he_path': he_path, 'real_path': real_path, 'fake_path': fake_path, 'slice':slice}
    
    def get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)

class HE_Dataset(Dataset):
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

class FrozenHE_Dataset(Dataset):
    def __init__(self, root_dir,slice_list):
        
        self.root_dir = root_dir
        self.transform = self.get_transform
        self.images = []
        self.img_paths = []
        self.slices = []

        # 遍历文件夹，收集图像文件路径和切片信息
        for slice_name in slice_list:
            root_dir1 = os.path.join(root_dir,slice_name)
            for root, _, files in os.walk(root_dir1):
                for file in files:
                    if file.endswith(".png") or file.endswith(".jpg"):
                        img_path = os.path.join(root, file)
                        slice_name = os.path.basename(root)
                        self.images.append(img_path)
                        self.img_paths.append(img_path)
                        self.slices.append(slice_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        slice_name = self.slices[idx]

        he_path = img_path
        he_img = Image.open(he_path).convert('RGB')
        # he_img = self.transform(he_img)
        transform = self.get_transform()
        he_img = transform(he_img)
        he_img = he_img
        he_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(he_img)
        return {'he': he_img, 'he_path': he_path, 'slice': slice_name}

    def get_transform(self, is_train=False):
        transform_list = []
        transform_list.append(transforms.Resize((512, 512)))

        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]

        transform = transforms.Compose(transform_list)  # 获取变换对象
        return transform
    
class autolabel_test_Dataset(Dataset): #测试一下最大池化好不好用
    def __init__(self, result_path):
        self.img_files = sorted(os.listdir(result_path))
        self.img_paths = [os.path.join(result_path, file_name) for file_name in self.img_files]
        self.transform = self.get_transform()
        
    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx):
        
        img_path = self.img_paths[idx]
        predict = int(os.path.basename(img_path).split('_')[0])
        
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        
        img = 1 - img #从这里开始吧
        
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        
        return {'img': img, 'predict':predict}
    
    def get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((256, 256)))

        transform_list += [transforms.ToTensor()]
        
        return transforms.Compose(transform_list)