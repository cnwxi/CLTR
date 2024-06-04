from __future__ import division

import os
import warnings

from config import return_args, args

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import get_root_logger, setup_seed
# import nni
# from nni.utils import merge_parameter
import time
import util.misc as utils
from utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch

from torch_npu.npu import amp # 导入AMP模块
from torch_npu.contrib import transfer_to_npu    # 使能自动迁移

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard
from tqdm import tqdm
# 准备backbone 特征提取网络，使用resnet
if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

os.environ["OMP_NUM_THREADS"] = "1"


def main(args):
    # 数据集准备
    if args['dataset'] == 'jhu':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'nwpu':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'
    elif args['dataset'] == "all":
        train_file = '/home/zshuai/AllForCLTR/train.npy'
        test_file = '/home/zshuai/AllForCLTR/val.npy'
    # 读取数据集npy（里面存储要读取的文件路径）
    with open(train_file, 'rb') as outfile:
        train_data = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_data = np.load(outfile).tolist()
    # 下面是分布式训练的设置
    utils.init_distributed_mode(return_args)
    # 构建模型
    model, criterion, postprocessors = build_model(return_args)
    # 分布式训练参数
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args['gpu_id']
    args['local_rank'] = int(os.environ["LOCAL_RANK"])
    if args['local_rank']==0:
        print(args)
    
    torch.cuda.set_device(args['local_rank'])
    # 模型转移到GPU
    model = model.cuda()
    if args['distributed']:
        #
        # torch.distributed.init_process_group(backend="hccl")
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']])
        path = './save_file/log_file/' + time.strftime(
            "%Y%m%d_%H%M", time.localtime(time.time()))
        args['save_path'] = path
        if (not os.path.exists(args['save_path'])) and args['local_rank'] == 0:
            os.makedirs(args['save_path'],exist_ok=True)
        if args['save']:
            logger = get_root_logger(path + '/1.log')
        else:
            logger = get_root_logger('./save_file/log_file/debug/debug.log')
        writer = SummaryWriter(path)

    else:
        args['train_patch'] = True
        return_args.train_patch = True
        model = nn.DataParallel(model, device_ids=[0])
        path = './save_file/log_file/debug/'
        args['save_path'] = path
        if not os.path.exists(args['save_path']):
            os.makedirs(path)
        logger = get_root_logger(path + 'debug.log')
        writer = SummaryWriter(path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params / 1e6)
    logger.info("model params: = {:.3f}\t".format(num_params / 1e6))

    optimizer = torch.optim.Adam([
        {
            'params': model.parameters(),
            'lr': args['lr']
        },
    ],
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    if args['local_rank'] == 0:
        logger.info(args)
    # 学习率调整
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args['lr_step']], gamma=0.1, last_epoch=-1)
    scaler = amp.GradScaler() 
    # 创建保存文件夹
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    # 如果有预训练模型，加载预训练模型
    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            print(f"=> loading checkpoint {args['pre']}")
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    print('best result:', args['best_pred'])
    logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] == 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(
            args['best_pred'], args['start_epoch']))

    if args['local_rank'] == 0:
        logger.info('start training!')

    # 开始训练
    for epoch in range(args['start_epoch'], args['epochs'] + 1):
        train(train_data, model, criterion, optimizer,scaler, epoch, scheduler,
              logger, writer, args)
        '''inference '''
        if (epoch+1) % args['test_per_epoch'] == 0 or epoch == 0:
            pred_mae, pred_mse, visi = validate(test_data, model, criterion,
                                                epoch, logger, args)
            
            writer.add_scalar('Metrcis/MAE', pred_mae, epoch)
            writer.add_scalar('Metrcis/MSE', pred_mse, epoch)
            # save_result
            if args['save']:
                is_best = pred_mae < args['best_pred']
                args['best_pred'] = min(pred_mae, args['best_pred'])
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args['pre'],
                        'state_dict': model.state_dict(),
                        'best_prec1': args['best_pred'],
                        'optimizer': optimizer.state_dict(),
                    }, visi, is_best, args['save_path'])

            end = time.time()

            if args['local_rank'] == 0:
                logger.info(
                    'Testing Epoch:[{}/{}]\t mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'
                    .format(epoch, args['epochs'], pred_mae, pred_mse,
                            args['best_pred']))


# 用于数据集的拼接
def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        #if return_args.train_patch:
        fname.append(item[0])

        for i in range(0, len(item[1])):
            imgs.append(item[1][i])

        for i in range(0, len(item[2])):
            targets.append(item[2][i])
        # else:
        # fname.append(item[0])
        # imgs.append(item[1])
        # targets.append(item[2])
    # torch.stack(imgs, 0)将list中的tensor拼接成一个tensor
    return fname, torch.stack(imgs, 0), targets


def train(Pre_data, model, criterion, optimizer,scaler, epoch, scheduler, logger,
          writer, args):
    # losses = AverageMeter()
    # torch.cuda.synchronize()作用：等待所有的cuda核心操作完成
    torch.cuda.synchronize()
    start = time.time()
    # 创建数据集对象
    train_data = dataset.listDataset(
        Pre_data,  # 数据集文件路径
        args['save_path'],  # 保存路径
        shuffle=True,  # 是否打乱数据
        transform=transforms.Compose([
            transforms.RandomGrayscale(
                p=args['gray_p'] if args['gray_aug'] else 0),  # 以概率p进行随机灰度
            transforms.ToTensor(),  # 转换为tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # 标准化
        ]),  # 数据预处理
        train=True,
        args=args)

    if args['distributed']:
        # DistributedSampler将数据集切分成多个子集，每个子集分配给不同的GPU
        datasampler = DistributedSampler(train_data,
                                         num_replicas=dist.get_world_size(),
                                         rank=args['local_rank'])
        # 设置当前epoch
        datasampler.set_epoch(epoch)
    else:
        datasampler = None
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args['batch_size'],
                                               drop_last=False,
                                               collate_fn=collate_wrapper,
                                               sampler=datasampler,
                                               num_workers=8,
                                               prefetch_factor=2,
                                               pin_memory=True)

    model.train()
    loss_log = []
    with tqdm(total=len(train_loader), ncols=100, desc=f'training｜epoch={epoch}') as pbar:
        for i, (fname, img, targets) in enumerate(train_loader):
            # print(fname)
            img = img.cuda()
            d6 = model(img)
            # 计算损失
            loss_dict = criterion(d6, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                       if k in weight_dict)
            # 记录损失
            writer.add_scalar('loss/total', loss,
                                len(train_loader) * epoch + i)
            writer.add_scalar('loss/loss_ce', loss_dict['loss_ce'],
                            len(train_loader) * epoch + i)
            writer.add_scalar('loss/loss_point', loss_dict['loss_point'],
                            len(train_loader) * epoch + i)
            writer.add_scalar('lr/lr_backbone',
                            optimizer.param_groups[0]['lr'],
                            len(train_loader) * epoch + i)

            loss_log.append(loss.item())
            # 梯度清零
            optimizer.zero_grad()
            # # 反向传播
            # loss.backward()
            # # 更新参数
            # optimizer.step()
            scaler.scale(loss).backward()    # loss缩放并反向转播
            scaler.step(optimizer)    # 更新参数（自动unscaling）
            scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
            pbar.update(1)
    # 等待所有的cuda核心操作完成
    torch.cuda.synchronize()
    epoch_time = time.time() - start
    scheduler.step()
    if args['local_rank'] == 0:
        logger.info(
            'Training Epoch:[{}/{}]\t loss={:.5f}\t lr={:.6f}\t epoch_time={:.3f}'
            .format(epoch, args['epochs'], np.mean(loss_log), args['lr'],
                    epoch_time))
    torch.cuda.empty_cache()


# 验证
def validate(Pre_data, model, criterion, epoch, logger, args):
    torch.cuda.synchronize()
    if args['local_rank'] == 0:
        logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    with tqdm(total=len(test_loader), ncols=100, desc=f'testing｜epoch={epoch}') as pbar:
        for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):

            if len(img.shape) == 5:
                img = img.squeeze(0)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            if len(kpoint.shape) == 5:
                kpoint = kpoint.squeeze(0)

            with torch.no_grad():
                img = img.cuda()
                outputs = model(img)

            out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
            prob = out_logits.sigmoid()
            prob = prob.view(1, -1, 2)
            out_logits = out_logits.view(1, -1, 2)
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                                kpoint.shape[0] * args['num_queries'], dim=1)
            count = 0
            gt_count = torch.sum(kpoint).item()
            for k in range(topk_values.shape[0]):
                sub_count = topk_values[k, :]
                sub_count[sub_count < args['threshold']] = 0
                sub_count[sub_count > 0] = 1
                sub_count = torch.sum(sub_count).item()
                count += sub_count

            mae += abs(count - gt_count)
            mse += abs(count - gt_count) * abs(count - gt_count)
            pbar.update(1)
    torch.cuda.synchronize()
    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))

    print('mae', mae, 'mse', mse)
    return mae, mse, visi


if __name__ == '__main__':
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(return_args, tuner_params))
    params= vars(return_args)
    main(params)
