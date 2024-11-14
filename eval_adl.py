import numpy as np
import torch
from modeling.netm1 import SemiADNet
from datasets import mvtecad_perlintest, visa_perlintest
import cv2
import os
from tqdm import tqdm
import shutil
import argparse
from utils import aucPerformance
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed

np.seterr(divide='ignore',invalid='ignore')



    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--dataset', type=str, default='mvtec', help="dataset name")
    parser.add_argument('--anomaly_source_path', type=str, default='./data/dtd/images', help="dataset anomaly source")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=256, help="the image size of input")
    #parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    #parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    #parser.add_argument('--criterion', type=str, default='deviation-focal', help="the loss function")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--cont", type=float, default=0.05, help="the percentage of contamination")
    #parser.add_argument("--gamma", type=float, default=0.5, help="gamma exponent")
    #parser.add_argument("--beta", type=float, default=0.8, help="beta percentage of focal-deviation loss")
    #parser.add_argument("--cmix_prob", type=float, default=0.4, help="cut_mix percentage for data augmentation")
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha-parameter for alpha divergence')
    parser.add_argument('--div_type',type=str, default='alpha', help='divergence type: alpha divergence')
    parser.add_argument('--lambda_hyp',type=float, default=0.1, help='hyperparameter controlling the radius on the divergence')
    parser.add_argument('--w_type', type=str, default='normalized', help='weight normalization: normalized/unnormalized')
    parser.add_argument('--report_name', type=str, default='result_report_mvtec_bal', help="name of the file where report will be stored")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.ramdn_seed)

    model = SemiADNet(args)
    model_wt_name = args.dataset +'_'  + args.classname + '_' + str(args.cont)  + '_' + args.weight_name
    print("model wt name:", model_wt_name)
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, model_wt_name)))
    model = model.cuda()
    model.eval()

    if args.dataset=='mvtec':
        test_set = mvtecad_perlintest.MVTecAD(args, train=False)
    elif args.dataset=='visa':
        test_set = visa_perlintest.VisaAD(args, train=False)
        
    kwargs = {'num_workers': args.workers}
        
    test_loader = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn_seed,
                                **kwargs)
                                
    tbar = tqdm(test_loader, desc='\r')
    #test_loss = 0.0
    total_target = np.array([])
    total_pred = list()
    total_pred_from_mask=list()
    anomaly_score_prediction = []
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            out_mask, outputs, prob = model(image.float())
            out_mask_sm = torch.softmax(out_mask, dim=1)
            

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1, padding=21 // 2).cpu().detach()
        out_mask_averaged = out_mask_averaged.view(int(out_mask_averaged.size(0)), -1)
        topk = max(int(out_mask_averaged.size(1) * 0.1), 1)
        image_score = torch.mean(out_mask_averaged, dim=1).view(-1,1)
        
        total_target = np.append(total_target, target.cpu().numpy())
        data = outputs.data.cpu().numpy()  
        total_pred = np.append(total_pred, data)
        total_pred_from_mask=np.append(total_pred_from_mask, image_score.numpy())
    
    roc, pr = aucPerformance(total_pred, total_target)
    with open(args.report_name + ".txt", 'a') as f:
        f.write("Class: %s, alpha: %.4f, lambda:%.4f, Contamination: %.4f,  AUC-ROC: %.4f, AUC-PR: %.4f \n" % (args.classname, args.alpha, args.lambda_hyp, args.cont, roc, pr))
