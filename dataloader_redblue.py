from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os

class MiniImagenet(Dataset): 
    def __init__(self, root_dir, r, transform, mode, color='red', pred=[], probability=[] ): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        pred = pred
        self.probability = probability
     
        if self.mode=='test':
            with open(self.root+'split/clean_validation') as f:            
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                img_path = 'validation/'+str(target) + '/'+img
                self.val_imgs.append(img_path)
                self.val_labels[img_path]=target                              
        else:    
            noise_file = '{}_noise_nl_{}'.format(color,r)
            with open(self.root+'split/'+noise_file) as f:
                lines=f.readlines()   
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                train_path = 'all_images/'
                train_imgs.append(train_path + img)
                self.train_labels[train_path + img]=target              
            if (self.mode == 'all') :
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [self.probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                  
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))  
                             
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif (self.mode=='all') or (self.mode=='pretext'):
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)   
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')   
            if self.transform is not None:
                img = self.transform(img)
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out
        
            
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


class red_miniImagenet32_dataloader():  
    def __init__(self, dataset, r, batch_size, num_workers, root_dir ):
        self.dataset = dataset
        self.r = r
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        
        if self.dataset=='mini_imagenet32_red':
            self.transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                ])   
        else:
            raise ValueError('Invalid dataset{}'.format(self.dataset)) 
        
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = MiniImagenet(dataset=self.dataset,  r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

                                     
        elif mode=='train':
            labeled_dataset = MiniImagenet(dataset=self.dataset,  r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = MiniImagenet(dataset=self.dataset,  r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",  pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_trainloader_map = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader, unlabeled_trainloader_map
        
        elif mode=='test':
            test_dataset = MiniImagenet(dataset=self.dataset, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = MiniImagenet(dataset=self.dataset, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader       


class StanfordCars(Dataset): 
    def __init__(self, root_dir, transform, meta_info, num_classes, color): 
        self.root = root_dir
        self.transform = transform
        self.mode = meta_info['mode']
        pred = meta_info['pred']
        num_class = num_classes
        self.probability = meta_info['probability']  
     
        if self.mode=='test':
            with open(self.root+'split/clean_validation') as f:            
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                img_path = 'validation/'+str(target) + '/'+img
                self.val_imgs.append(img_path)
                self.val_labels[img_path]=target                              
        else:    
            noise_file = '{}_noise_nl_{}'.format(color,meta_info['noise_rate'])
            with open(self.root+'split/'+noise_file) as f:
                lines=f.readlines()   
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                train_path = 'all_images/'
                train_imgs.append(train_path + img)
                self.train_labels[train_path + img]=target              
            if (self.mode == 'all') or (self.mode == 'neighbor') or (self.mode=='pretext'):
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [self.probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                  
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif (self.mode=='all') or (self.mode=='pretext'):
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            img = Image.open(self.root+img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)   
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            img = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            if self.transform is not None:
                img = self.transform(img)
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out
        elif self.mode=='neighbor':
            img_path = self.train_imgs[index]
            img = Image.open(self.root+img_path).convert('RGB')
            target = self.train_labels[img_path]
            if self.transform is not None:
                img = self.transform(img)
            out = {'image': img, 'target': target, 'meta': {'index': index}}
            return out
        
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    