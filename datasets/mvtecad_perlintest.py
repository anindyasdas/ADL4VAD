import numpy as np
import PIL
import os, sys, shutil
from PIL import Image
from torchvision import transforms
from copy import deepcopy
import cv2
import glob
import torch
import random
import imgaug.augmenters as iaa


from datasets.base_dataset import BaseADDataset

from modeling.transformations.perlin import rand_perlin_2d_np

class MVTecAD(BaseADDataset):

    def __init__(self, args, train = True):
        super(MVTecAD).__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname
        self.anomaly_source_paths = sorted(glob.glob(self.args.anomaly_source_path+"/*/*.jpg"))
        print(self.args.anomaly_source_path, len(self.anomaly_source_paths))
        

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.anomaly_train_dirname= 'anomaly_train_gen_samples' #os.path.join('..', 'anomaly_train_gen_samples')
        
        self.transform = [self.transform_train_normal, self.transform_train_outlier] if self.train else [self.transform_test, self.transform_test]

        
        normal_data = list()
        if self.train:
            split = 'train'
        else:
            split = 'test'
        normal_files = os.listdir(os.path.join(self.root, split, 'good'))
        for file in normal_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_data.append(split + '/good/' + file)
                
        #####compute contamination_size from training data_size, contamination used in training ,not to be used for evaluation during testing########
        normal_train_data = list()
        normal_train_files = os.listdir(os.path.join(self.root, 'train', 'good'))
        for file in normal_train_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_train_data.append('train' + '/good/' + file)
        self.contamination_size= int((self.args.cont/(1-self.args.cont))*len(normal_train_data))
        print("cont size", self.contamination_size, self.args.cont)
        ###########################################################################################
        
        if split=='train':
            anomaly_data=self.get_outlier()
            print(split)
            print("normal data:", len(normal_data))
            normal_data=normal_data+anomaly_data
            print("contaminated normal data:", len(normal_data))
            outlier_data= deepcopy(normal_data) #Copying normal data to be used for outliers by augmentation
            print("pseudo outlier_data:",  len(outlier_data))
        else:
            print(split)
            print("normal data:",  len(normal_data))
            outlier_data = self.get_outlier()
            print("outlier_data:", len(outlier_data))
            #print("len test outlier", len(outlier_data))
        outlier_data.sort()

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.images = normal_data + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()
        
        
        ######augmentation####
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        #sequential augmentation
        self.augmentation = iaa.Sequential([iaa.Flipud(p=0.5), # flip vertically with a probability of 0.5
                      iaa.Fliplr(p=0.5),  # flip horizontally with a probability of 0.5
                      iaa.Affine(rotate=(15, 75)),
                      iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=20))])

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.brightness_contrast_aug=iaa.Sequential([iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30))])
        self.saturation_hue_aug=iaa.AddToHueAndSaturation((-20,20),per_channel=True)
        
        self.transform_img2tensor = transforms.Compose([
            transforms.ToTensor()])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    
            
    
        
    
    


    def get_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        outlier_data = list()
        for cl in outlier_classes:
            if cl == 'good':
                continue
            outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    outlier_data.append('test/' + cl + '/' + file)
        
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        
        if self.train:
            print("available total anomalies:", len(outlier_data))
            outlier_samples= self.synthetic_contamination(outlier_data, self.contamination_size)
            return outlier_samples
        else:
            return outlier_data
    
    def synthetic_contamination(self, anorm, num_contamination):
        print("Synthetic anomaly contamination .....")
        anorm_img = [self.load_image(os.path.join(self.root, img_path)) for img_path in anorm]
        #concat accross new axis
        anorm_img= np.stack(anorm_img, axis=0)
        #print(anorm_img.shape)
        try:
            idx_contamination = np.random.choice(np.arange(anorm_img.shape[0]),num_contamination,replace=False)
        except:
            idx_contamination = np.random.choice(np.arange(anorm_img.shape[0]),num_contamination,replace=True)
        train_contamination = anorm_img[idx_contamination]
        
        train_contamination = train_contamination + np.random.randn(*train_contamination.shape)*2.0
        outlier_path=[]
        dirpath= os.path.join(self.root, self.anomaly_train_dirname)
        self.delete_temp_directory(dirpath)
        self.ensure_directory_exists(dirpath)
        for idx in range(0, train_contamination.shape[0]):
            img_data= train_contamination[idx]
            img_data=self.get_normalized_img(img_data)
            #img_data=self.random_rotation(img_data)
            #print(img_data.shape)
            save_path=os.path.join(self.anomaly_train_dirname, str(idx) + '.jpg')
            outlier_path.append(save_path)
            self.save_image(img_data, os.path.join(self.root, save_path))
        return outlier_path
    
    def delete_temp_directory(self, dirpath):
        #delete the already existing directory
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            print(f"Deleting old directory '{dirpath}' .")
            shutil.rmtree(dirpath)
    
    def ensure_directory_exists(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
        
        

    def load_image(self, path):
        image = cv2.imread(path)
        # Convert BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, dsize=(self.args.img_size,self.args.img_size))
        #returns numpy array
        return image
        
    def random_rotation(self, image):
        p = random.random()
        #random flip while saving
        if p<=0.33: #horizontal
            image = cv2.flip(image, 0) 
        elif p>0.33 and p<=0.66: #vertical
            image = cv2.flip(image, 1)
        return image
        
        
    def save_image(self, image, path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        
    def get_normalized_img(self, img_array):
        min_val= np.min(img_array)
        max_val= np.max(img_array)
        img_array=((img_array-min_val)*255.0)/(max_val-min_val)
        return img_array.astype('uint8')
        
    

    def transform_train_normal(self, image, anomaly_source_img=None):
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        anomaly_mask=np.zeros((image.shape[0], image.shape[1],1))
        
        return self.transform_img2tensor(image), self.transform_img2tensor(anomaly_mask)
    
    
        
    def transform_train_outlier(self, image, anomaly_source_img):
        
        
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        
        
        #(Augmentation using perlin Noise)
        if do_aug_orig:
            image = self.rot(image=image)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_img)
        
        augmented_image1=cv2.cvtColor(augmented_image*255.0, cv2.COLOR_RGB2BGR)
        anomaly_mask1=cv2.cvtColor(anomaly_mask*255.0, cv2.COLOR_RGB2BGR)
        anomaly_source_img1=cv2.cvtColor(anomaly_source_img, cv2.COLOR_RGB2BGR)
        return self.transform_img2tensor(augmented_image), self.transform_img2tensor(anomaly_mask)
        
        
    

    def transform_test(self, image, anomaly_source_img=None):
        image = image / 255.0
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        anomaly_mask=np.zeros((image.shape[0], image.shape[1],1))
        
        return self.transform_img2tensor(image), self.transform_img2tensor(anomaly_mask)
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def augment_image(self, image, anomaly_source_img):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.args.img_size, self.args.img_size), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        
        if no_anomaly < 0.0:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
        
   
    def __len__(self):
        return len(self.images)

    def getitem(self, index):
        transform_norm, transform_outlier = self.transform
        #transform_norm_compare, transform_outlier_compare = self.transform_compare
        image = self.load_image(os.path.join(self.root, self.images[index]))
        
        if self.labels[index]==0:
            #print("transforming normal", transform_norm)
            #add image compare , for comparision of two variation of same images and maximize the difference
            tr_image, tr_mask=transform_norm(image)
            image=PIL.Image.fromarray(image.astype('uint8'))
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'raw_image': image, 'seg_label': None}
        else:
            #print("transforming outlier", transform_outlier)
            
            if self.train:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_source_img=self.load_image(self.anomaly_source_paths[anomaly_source_idx])
                image_label= None
            else:
                anomaly_source_img=None
                image_label = self.load_image(os.path.join(self.root,self.images[index]).replace('test','ground_truth').replace('.png','_mask.png'))
                #print("image_label", image_label.shape)
                image_label=PIL.Image.fromarray(image_label.astype('uint8'))
                #print("image_label", np.array(image_label).shape)
            tr_image, tr_mask=transform_outlier(image, anomaly_source_img)
            image=PIL.Image.fromarray(image.astype('uint8'))
            
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'raw_image': image, 'seg_label':image_label}
        return sample
        
    def __getitem__(self, index):
        transform_norm, transform_outlier = self.transform
        #transform_norm_compare, transform_outlier_compare = self.transform_compare
        image = self.load_image(os.path.join(self.root, self.images[index]))
        
        if self.labels[index]==0:
            #print("transforming normal", transform_norm)
            #add image compare , for comparision of two variation of same images and maximize the difference
            tr_image, tr_mask=transform_norm(image)
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index]}
        else:
            #print("transforming outlier", transform_outlier)
            
            if self.train:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_source_img=self.load_image(self.anomaly_source_paths[anomaly_source_idx])
            else:
                anomaly_source_img=None
            tr_image, tr_mask=transform_outlier(image, anomaly_source_img)
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index]}
        return sample

    
