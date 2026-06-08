import os
import json
import argparse
import torch
import dataloaders
import models
import math
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import  CE_loss
# from utils.mmd_loss import MMD_loss
# from utils.infoNCE_loss import SupDistLoss, SoftDistLoss
torch.set_num_threads(2)



def main(config, method, backbone, threshold, resume, gpu, aug_type='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.manual_seed(42)

    config['model']['method'] = method
    config['model']['backbone'] = backbone
    config['model']['confidence_thr'] = str(threshold)
    config['val_loader']['aug_type']  = aug_type
    config['train_source']['aug_type'] = aug_type
    config['train_target']['aug_type'] = aug_type

    backbone = config['model']['backbone'] 
    config['name'] = config['name'].replace('source', config['source'])\
                                   .replace('target', config['target'])
    config['experim_name'] = config['experim_name'].replace('source', config['source'])\
                                                   .replace('target', config['target'])\
                                                   .replace('method', method)

    config['train_source']['data_dir'] = config['train_source']['data_dir'].replace('source', config['source'])
    config['train_target']['data_dir'] = config['train_target']['data_dir'].replace('target', config['target'])
    config['val_loader']['data_dir'] = config['val_loader']['data_dir'].replace('target', config['target'])
    config['trainer']['save_dir'] = config['trainer']['save_dir'].replace('source', config['source'])\
                                                                 .replace('target', config['target'])\
                                                                 .replace('backbone', backbone)\
                                                                 .replace('method', method)
    config['trainer']['log_dir'] = config['trainer']['log_dir'].replace('source', config['source'])\
                                                               .replace('target', config['target'])\
                                                               .replace('backbone', backbone)

    # DATA LOADERS
    source_loader = dataloaders.CDDataset(config['train_source'])
    target_loader = dataloaders.CDDataset(config['train_target'])

    print (config)
    print ('source: ', len(source_loader))
    print ('target: ', len(target_loader))
    val_loader = dataloaders.CDDataset(config['val_loader'])
    iter_per_epoch = len(target_loader)

    # MODEL
    if backbone == 'ResNet50':
        model = models.ResNet50_CD(num_classes=val_loader.dataset.num_classes, conf=config)
    elif backbone == 'HRNet':
        model = models.HRNet_CD(num_classes=val_loader.dataset.num_classes, conf=config)
    elif backbone == 'SegFormer':
        model = models.SegFormer_CD(num_classes=val_loader.dataset.num_classes, conf=config)
    elif backbone == 'ResNet101':
        model = models.ResNet101_CD(num_classes=val_loader.dataset.num_classes, conf=config)
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch)
    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_source_to_target.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-s', '--source', default='LEVIR', type=str,
                        help='Source dataset')    
    parser.add_argument('-t', '--target', default='GZ', type=str,
                        help='Target dataset')    
    parser.add_argument('-m', '--method', default='DANN+B2ST', type=str,
                        help='test method')
    parser.add_argument('-b', '--backbone', default='ResNet50', type=str,
                        help='selected encoder backbone')
    parser.add_argument('-thr', '--threshold', default=0.95, type=float,
                        help='threshold')
    parser.add_argument('-r', '--resume', default='./saved/source_to_target/backbone/method/checkpoint_thr-threshold.pth', type=str,
                        help='Path to the .pth model checkpoint to resume training')
    # parser.add_argument('-r', '--resume', default=None, type=str,
    #                     help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    print (args.config)
    config = json.load(open(args.config.replace('source', args.source)\
                                       .replace('target', args.target)))
    if args.resume != None:
        args.resume = args.resume.replace('method', args.method)\
                                 .replace('backbone', args.backbone)\
                                 .replace('source', args.source)\
                                 .replace('target', args.target)\
                                 .replace('threshold', str(args.threshold))
    args.resume = args.resume if os.path.exists(args.resume) else None
    main(config, args.method, args.backbone, args.threshold, args.resume, str(args.gpu))
