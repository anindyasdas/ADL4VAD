from datasets import mvtecad_perlintest, visa_perlintest
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler
import numpy as np

def _init_fn(worker_id):
    seed=42
    np.random.seed(int(seed))

def build_dataloader(args, **kwargs):
    if args.dataset=='mvtec':
        train_set = mvtecad_perlintest.MVTecAD(args, train=True)
        test_set = mvtecad_perlintest.MVTecAD(args, train=False)
    elif args.dataset=='visa':
        train_set = visa_perlintest.VisaAD(args, train=True)
        test_set = visa_perlintest.VisaAD(args, train=False)
    train_loader = DataLoader(train_set,
                                worker_init_fn=worker_init_fn_seed,
                                batch_sampler=BalancedBatchSampler(args, train_set),
                                **kwargs)
    test_loader = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn_seed,
                                **kwargs)
    return train_loader, test_loader