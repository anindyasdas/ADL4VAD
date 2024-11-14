#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch
import os
import random
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;
    


def seed_everything(seed=42):
    """"
    Seed everything.
    """
    random.seed(seed) #python
    cv2.setRNGSeed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    np.random.seed(seed) #numpy
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) #check if any not deterministic process
    
def run_k_means(score):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(score.detach().cpu().numpy())
    centroids = kmeans.cluster_centers_
    min_centroid_index = np.argmin(centroids.flatten())
    labels = kmeans.labels_
    nor_index =np.argwhere(labels == min_centroid_index).flatten()
    cluster_labels = np.ones(len(score))
    cluster_labels[nor_index]=0
    cluster_labels=torch.from_numpy(cluster_labels)
    return cluster_labels

def get_loss_weights(losses, div_type, alpha, lambda_hyp, w_type, iteration,
                     burnin):
    """Compute weights for reweighing instance losses."""
    if iteration <= burnin or div_type == 'none':
        weights = np.ones_like(losses)
    elif div_type == 'alpha':
        if np.abs(alpha - 1.) < 1e-3:
            losses = torch.tensor(losses)
            weights = torch.exp(-1 * losses / lambda_hyp)
            weights = weights.numpy()
        else:
            weights = np.power(np.maximum((1. - alpha) * losses + lambda_hyp, 0.0), 
                      1. / (alpha - 1.))
    else:
        raise NotImplementedError(
          'Divergence {} is not implemented'.format(div_type))
    
    if w_type == 'normalized':
        weights = weights / np.sum(weights)  #  * len(labels)
    return weights
    
def rand_bbox_(size, lam):
    x1= size[0]
    y1= size[1]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    #print(lam, cut_rat, cut_w, cut_h)

    # uniform
    cx = np.random.randint(x1,x1+W)
    cy = np.random.randint(y1,y1+H)
    #print("LL:", cx, cy, cut_w, cut_h, W, H)

    bbx1 = np.clip(cx - cut_w // 2, x1, x1+W)
    bby1 = np.clip(cy - cut_h // 2, y1, y1+H)
    bbx2 = np.clip(cx + cut_w // 2, x1, x1+W)
    bby2 = np.clip(cy + cut_h // 2, y1, y1+H)
    
    print("LL:", cx, cy, cut_w, cut_h, W, H, [bbx1, bby1, bbx2, bby2])
    return bbx1, bby1, bbx2, bby2
    
    
def rand_bbox(size, lam):
    x1= size[0]
    y1= size[1]
    W = size[2]
    H = size[3]
    x2=x1+W
    y2=y1+H
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    #handling when height or width of the patch is 0, set to default size of 2
    if cut_h==0:
        cut_h+=2
    if cut_w==0:
        cut_w+=2
    
    #print(lam, cut_rat, cut_w, cut_h)

    # uniform
    cx = np.random.randint(x1,x2-cut_w+1)
    cy = np.random.randint(y1,y2-cut_h+1)
    #print("LL:", cx, cy, cut_w, cut_h, W, H)
        
    
    
    #print("LL:", cx, cy, cut_w, cut_h, W, H, [bbx1, bby1, bbx2, bby2])
    return cx, cy, cx + cut_w, cy + cut_h
    
    
def rand_bbox1(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = int(random.uniform(0, W - cut_w)) # np.random.randint(W)
    cy = int(random.uniform(0, H - cut_h)) #np.random.randint(H)
    box1 = [cx, cy, cx + cut_w, cy + cut_h]
    r = np.random.rand(1)
    if r < 0.5:
        cx1 = int(random.uniform(0, W - cut_w)) # np.random.randint(W)
        cy1 = int(random.uniform(0, H - cut_h)) #np.random.randint(H)
        box2 = [cx1, cy1, cx1 + cut_w, cy1 + cut_h]
    else:
        box2=box1


    return box1, box2
    
    
def compare_bbox(list1,list2, area=0.2):
    #this method compare two bounding boxes, if IOU is greater than 0.2 it returns 
    # intersectection else it returns the larger bounding box
    #print(x1,y1,x2,y2,x1_,y1_,x2_,y2_)
    x1,y1,x2,y2=list1
    x_1,y_1,x_2,y_2=list2
    nx1=max(x1,x_1)
    ny1=max(y1,y_1)
    nx2=min(x2,x_2)
    ny2=min(y2,y_2)
    area1=(y2-y1)*(x2-x1)
    area2=(y_2-y_1)*(x_2-x_1)
    if area2>area1:
        default=list2
    else:
        default=list1
    
    if ny2>ny1 and nx2>nx1:
        intersect= (ny2-ny1)*(nx2-nx1)
        uni=  area1+area2-intersect
        iou=intersect/uni
        if iou >area:
            return [nx1,ny1,nx2,ny2]
        else:
            return default
    else:
        return default


