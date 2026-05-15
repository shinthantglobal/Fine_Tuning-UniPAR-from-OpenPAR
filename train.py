import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import get_multi_dataset
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus, load_ckpt
from solver import make_optimizer

from solver.scheduler_factory import create_scheduler

set_seed(605)

def main(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # 将--gpus参数映射到环境变量
        print(f"Using GPU(s): {args.gpus}")
    if args.resume:
        log_dir = args.resume
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Resume directory not found: {log_dir}")
        print(f"Resuming training in existing log directory: {log_dir}")
    else:
        start_time = time_str()
        print(f'start_time is {start_time}')
        log_dir = os.path.join('logs', args.save_place)
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.join(log_dir, start_time)
        os.makedirs(log_dir, exist_ok=True)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    multi_train_set, multi_valid_set, criterion_dict = get_multi_dataset(args)

    # Debugg:
    print(f"-> Debugg: {len(multi_train_set)}, {len(multi_valid_set)}, {criterion_dict} <-")

    print(dir(multi_train_set))  # 输出所有属性和方法
    train_loader = DataLoader(
        dataset=multi_train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=multi_valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = TransformerClassifier(args=args)

    if torch.cuda.is_available():
        model = model.cuda()


    trainer(epoch=args.epoch,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion_dict=criterion_dict,
            path=log_dir,
            args=args)
    
def trainer(epoch, model, train_loader, valid_loader, criterion_dict, path, args):
    if isinstance(args.dataset, str):
       datasets = [args.dataset]
    else:
        datasets = args.dataset
    
    # Check for resume
    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    start_dataset_idx = 0
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        loaded_epoch, loaded_dataset, loaded_metric = load_ckpt(model, None, None, checkpoint_path)
        if loaded_dataset in datasets:
            start_dataset_idx = datasets.index(loaded_dataset)
            start_epoch = loaded_epoch + 1
            print(f"Resuming from dataset {loaded_dataset}, epoch {loaded_epoch}")
        else:
            print("Checkpoint dataset not in current datasets, starting from beginning")
    else:
        print("No checkpoint found, starting from beginning")
    
    for dataset_idx, dataset in enumerate(datasets):
        if dataset_idx < start_dataset_idx:
            continue
        print(f'================================', dataset, '================================', '\n')
        torch.cuda.empty_cache()
        optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = create_scheduler(optimizer, num_epochs=args.epoch, lr=args.lr, warmup_t=5)
        
        # If resuming and this is the resuming dataset, load optimizer and scheduler states
        if dataset_idx == start_dataset_idx and start_epoch > 1:
            load_ckpt(model, optimizer, scheduler, checkpoint_path)
        
        criterion = criterion_dict[dataset]
        train_loader.dataset.init_set(dataset)
        valid_loader.dataset.init_set(dataset)
        if dataset=='DUKE': 
            total_epochs = args.epoch * 5
        else:
            total_epochs = args.epoch
        
        epoch_start = start_epoch if dataset_idx == start_dataset_idx else 1
        
        for i in range(epoch_start, total_epochs+1):
            scheduler.step(i)
            train_loss, train_gt, train_probs = batch_trainer(
                epoch=i,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                args=args
            )

            valid_loss, valid_gt, valid_probs = valid_trainer(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                args=args
            )
            train_result = get_pedestrian_metrics(train_gt, train_probs)

            print(f'Evaluation on train set, \n',
                  'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                      train_result.ma, np.mean(train_result.label_pos_recall), np.mean(train_result.label_neg_recall)),
                  'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                      train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                      train_result.instance_f1))
            
            valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

            print(f'Evaluation on test set, \n',
                  'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                      valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
                  'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                      valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                      valid_result.instance_f1))

            print('-' * 60)
            print(f"{path}!")
            # Save checkpoint
            save_ckpt(model, optimizer, scheduler, checkpoint_path, i, dataset, valid_result)
            # Also save a timestamped version
            save_ckpt(model, optimizer, scheduler, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, dataset, valid_result)
            print(f"!!! MISSION SUCCESS: Final model !!!")

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()