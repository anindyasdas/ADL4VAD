import numpy as np
import torch
import torch.nn as nn
import random
import argparse
import os
from copy import deepcopy
from scipy import interpolate
from sklearn.cluster import KMeans, MiniBatchKMeans
from dataloaders.dataloadert import build_dataloader
from modeling.netm1 import SemiADNet
from tqdm import tqdm
from utils import aucPerformance, run_k_means, get_loss_weights, seed_everything
from loss import FocalLoss

import warnings
warnings.filterwarnings('ignore')







    
class SoftDeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def get_losses(self, y_pred):
        confidence_margin = 10.
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return inlier_loss, outlier_loss
    

    def forward(self, y_pred, prob):
        inlier_loss, outlier_loss = self.get_losses(y_pred)
        total_loss= (1-prob.detach())*inlier_loss+ (prob.detach())*outlier_loss
        return total_loss
        
##############################################################################       ############### 
class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kwargs)

        self.model = SemiADNet(args)
        
        
        
        #Code for instance re-weighting
        self.total_iter= len(self.train_loader) *self.args.epochs
        #burn-in (initial training with uniform weight) 
        self.burnin= int(self.total_iter *0.06)
        self.curr_iter=0
        self.burnin_interp_fn = interpolate.interp1d([self.burnin, self.burnin *3, self.total_iter],
                      [self.args.lambda_hyp * 10, self.args.lambda_hyp, self.args.lambda_hyp])
        

        self.criterion= SoftDeviationLoss()
        self.criterion2= nn.BCELoss(reduce=False)
        self.criterion3= FocalLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()
           self.criterion2=self.criterion2.cuda()
           self.criterion3=self.criterion3.cuda()
    

    def train(self, epoch):
        train_loss = 0.0
        train_segment_loss=0.0
        train_abnormal_loss=0.0
        train_ce_loss=0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        if epoch >=0:
            self.correct_label=False
            
        for i, sample in enumerate(tbar):
            image, anomaly_mask, target = sample['image'], sample['mask'], sample['label']
            
            if self.args.cuda:
                image, anomaly_mask, target = image.cuda(), anomaly_mask.cuda(), target.cuda() 
            
            out_mask, outputs, prob = self.model(image)
            
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            
                
            
            segment_loss = self.criterion3(out_mask_sm, anomaly_mask)
            
            #########approximation of lambda_hyp based on iteration###############
            if self.burnin > 0 and self.curr_iter > self.burnin:
                lambda_hyp = self.burnin_interp_fn(self.curr_iter)
                losses=self.criterion(outputs, prob.detach().float()).view(-1, 1)
                cluster_labels=run_k_means(outputs).cuda()
                
                combined_labels= (cluster_labels + target).cpu()
                nor_index =np.argwhere(combined_labels>= 1).flatten()
                combined_labels[nor_index]=1
                combined_labels=combined_labels.cuda()
                
                if np.random.rand() >=0.4:
                    CE_losses= self.criterion2(prob.squeeze(), combined_labels.float())
                else:
                    CE_losses= self.criterion2(prob.squeeze(), target.float())
                
                
                
               
                
            else:
                lambda_hyp = self.args.lambda_hyp
                losses=self.criterion(outputs, target.unsqueeze(1).float()).view(-1, 1)
                CE_losses= self.criterion2(prob.squeeze(), target.float())
                
            
            
            
            batch_size=CE_losses.size()[0]
            
            
            loss_weights1= torch.from_numpy(get_loss_weights(CE_losses.detach().cpu().numpy(), self.args.div_type, self.args.alpha, 
                        self.args.lambda_hyp, self.args.w_type, self.curr_iter,
                     self.burnin))
            loss_weights2= torch.from_numpy(get_loss_weights(losses.detach().cpu().numpy(), self.args.div_type, self.args.alpha, 
                        self.args.lambda_hyp, self.args.w_type, self.curr_iter,
                     self.burnin))
            
            if self.args.cuda:
                loss_weights1=loss_weights1.cuda()
                loss_weights2=loss_weights2.cuda()

            CE_loss=torch.mean(CE_losses*loss_weights1)
            loss=torch.mean(losses*loss_weights2)
            self.optimizer.zero_grad()
            total_loss= loss + CE_loss + segment_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.curr_iter+=1
           
            train_segment_loss += segment_loss.item()
            train_abnormal_loss += loss.item()
            train_loss +=loss.item()
            train_ce_loss += CE_loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.5f Train segment loss: %.5f Train loss ce: %.5f' % (epoch, train_loss / (i + 1), train_segment_loss / (i + 1), train_ce_loss / (i + 1)))
        self.scheduler.step()

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_target = np.array([])
        total_pred = list()
        total_pred_from_mask=list()
        anomaly_score_prediction = []
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                
                out_mask, outputs, prob = self.model(image.float())
                
                out_mask_sm = torch.softmax(out_mask, dim=1)
                losses=self.criterion(outputs, prob.detach().float()).view(-1, 1)
                
                loss=torch.mean(losses)

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach()
            out_mask_averaged = out_mask_averaged.view(int(out_mask_averaged.size(0)), -1)
            topk = max(int(out_mask_averaged.size(1) * 0.1), 1)
            image_score = torch.mean(out_mask_averaged, dim=1).view(-1,1)
            
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_target = np.append(total_target, target.cpu().numpy())
            
            data = outputs.data.cpu().numpy()
            
            total_pred = np.append(total_pred, data)
            total_pred_from_mask=np.append(total_pred_from_mask, image_score.numpy())
        roc, pr = aucPerformance(total_pred, total_target)
        with open(self.args.report_name + ".txt", 'a') as f:
            f.write("Class: %s, alpha: %.4f, lambda:%.4f, Contamination: %.4f,  AUC-ROC: %.4f, AUC-PR: %.4f \n" % (self.args.classname, self.args.alpha, self.args.lambda_hyp, self.args.cont, roc, pr))
        return roc, pr

    def save_weights(self, filename):
        model_wt_name = self.args.dataset +'_' +self.args.classname + '_' + str(self.args.cont) +'_'+ filename
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, model_wt_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--dataset', type=str, default='mvtec', help="dataset name")
    parser.add_argument('--anomaly_source_path', type=str, default='./data/dtd/images', help="dataset anomaly source")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=256, help="the image size of input")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    #parser.add_argument('--criterion', type=str, default='deviation-focal', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--cont", type=float, default=0.05, help="the percentage of contamination")
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha-parameter for alpha divergence')
    parser.add_argument('--div_type',type=str, default='alpha', help='divergence type: alpha divergence')
    parser.add_argument('--lambda_hyp',type=float, default=0.1, help='hyperparameter controlling the radius on the divergence')
    parser.add_argument('--w_type', type=str, default='normalized', help='weight normalization: normalized/unnormalized')
    parser.add_argument('--report_name', type=str, default='result_report_mvtec_bal', help="name of the file where report will be stored")
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.ramdn_seed)
    trainer = Trainer(args)
    
    


    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("class:", args.classname)
    for epoch in range(0, trainer.args.epochs):
        trainer.train(epoch)
    trainer.eval()
    trainer.save_weights(args.weight_name)

