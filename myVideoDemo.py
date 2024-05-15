from __future__ import division

import os
import warnings
from collections import OrderedDict
from config import return_args, args
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from utils import setup_seed
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from Networks.CDETR import build_model
from tqdm import tqdm
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

def resize_w_h(h, w):
    max_size = 2048
    min_size = 384
    im_h, im_w = h, w
    rate = 1.0 * max_size / im_h
    rate_w = im_w * rate
    if rate_w > max_size:
        rate = 1.0 * max_size / im_w
    tmp_h = int(1.0 * im_h * rate / 16) * 16

    if tmp_h < min_size:
        rate = 1.0 * min_size / im_h
    tmp_w = int(1.0 * im_w * rate / 16) * 16

    if tmp_w < min_size:
        rate = 1.0 * min_size / im_w
    tmp_h = min(max(int(1.0 * im_h * rate / 16) * 16, min_size), max_size)
    tmp_w = min(max(int(1.0 * im_w * rate / 16) * 16, min_size), max_size)

    rate_h = 1.0 * tmp_h / im_h
    rate_w = 1.0 * tmp_w / im_w
    assert tmp_h >= min_size and tmp_h <= max_size
    assert tmp_w >= min_size and tmp_w <= max_size
    return tmp_w, tmp_h

def show_map(out_pointes, frame, width, height, crop_size, num_h, num_w):
    kpoint_list = []
    confidence_list = []

    for i in range(len(out_pointes)):

        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:,
                                              1:3].data.cpu().numpy().tolist()
        k = np.zeros((crop_size, crop_size))
        c_map = np.zeros((crop_size, crop_size))
        '''get coordinate'''
        for j in range(len(out_point)):
            if out_value[j] < 0.25:
                break
            x = int(out_point[j][0])
            y = int(out_point[j][1])
            k[x, y] = 1

        kpoint_list.append(k)
        confidence_list.append(c_map)

    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(
        0, 2, 1, 3).contiguous().view(num_h, crop_size,
                                      width).view(height, width).cpu().numpy()
    density_map = gaussian_filter(kpoint.copy(), 6)
    density_map = density_map / np.max(density_map) * 255
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    '''obtain the coordinate '''
    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])

    point_map = np.zeros((int(kpoint.shape[0]), int(kpoint.shape[1]), 3),
                         dtype="uint8") + 255  # 22
    for i in range(count):
        w = int(pred_coor[1][i])
        h = int(pred_coor[0][i])

        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
        cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)

    return point_map, density_map, frame, count
def process_video(video_path, model,query_num,step=5):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    # width, height = resize_w_h(height, width)
    # print(width, height)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    '''out video'''
    # width = 1792
    # height = 1024
    width = 1024
    height = 768
    print(width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path_split = video_path.split('/')[:-2]
    output_path = '/'.join(output_path_split) + f'/cltroutput/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('mkdir output path')
    output_name = video_path.split('/')[-1].split('.')[0] + '.mp4'
    output_path = output_path + output_name
    print(output_path)
    out = cv2.VideoWriter(f"{output_path}", fourcc, 30,
                          (width , height))
    history = []
    cnt=0
    num_total=frames//step

    with tqdm(total=num_total,ncols=50) as pbar:
        while True:
            try:
                ret, frame = cap.read()
                
            except:
                print("test end")
                cap.release()
                break
            if not ret:
                    break
            cnt+=1
            if cnt%step==0:
                frame = cv2.resize(frame, (width, height))
                frame = frame.copy()
                ori_frame = frame.copy()
                image = tensor_transform(frame)
                image = img_transform(image)

                width, height = image.shape[2], image.shape[1]
                num_w = int(width / 256)
                num_h = int(height / 256)
                image = image.view(3, num_h, 256,
                                width).view(3, num_h, 256, num_w, 256)
                image = image.permute(0, 1, 3, 2, 4).contiguous().view(
                    3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)

                with torch.no_grad():
                    image = image.cuda()
                    outputs = model(image)

                    out_logits, out_point = outputs['pred_logits'], outputs[
                        'pred_points']

                    prob = out_logits.sigmoid()
                    topk_values, topk_indexes = torch.topk(prob.view(
                        out_logits.shape[0], -1),query_num,dim=1)

                    topk_points = topk_indexes // out_logits.shape[2]
                    out_point = torch.gather(
                        out_point, 1,
                        topk_points.unsqueeze(-1).repeat(1, 1, 2))
                    out_point = out_point * 256

                    value_points = torch.cat([topk_values.unsqueeze(2), out_point],
                                            2)
                    crop_size = 256
                    kpoint_map, density_map, frame, count = show_map(
                        value_points, frame, width, height, crop_size, num_h,
                        num_w)

                    # res1 = np.hstack((ori_frame, kpoint_map))
                    # res2 = np.hstack((density_map, frame))
                    # res = np.vstack((res1, res2))
                    res =cv2.addWeighted(frame, 0.6, density_map, 0.4, 0)
                    history.append(count)
                    if len(history) >= 50:
                        history.pop(0)
                    history_mean = round(np.mean(history))
                    cv2.putText(res, f"Count:{history_mean}", (80, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                    out.write(res)
                    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    # cv2.imshow('result', res)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                    pbar.update(1)
        # cv2.destroyAllWindows()
        cap.release()
        out.release()
def main(args):
    # python video_demo.py --video_path ./video_demo/demo.mp4 --num_queries 700 --pre video_model.pth
    args['video_path'] = '/media/ubuntu/2.0TB/wxy/talkweb_data/data'
    args['num_queries'] = 700
    args['pre'] = 'video_model.pth'
    step=5
    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    torch.cuda.set_device(0)
    model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    if args['pre']:
        if os.path.isfile(args['pre']):
            checkpoint = torch.load(args['pre'])['state_dict']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # if 'backbone' in k or 'transformer' in k:
                name = k.replace(
                    'bbox', 'point'
                )  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                new_state_dict[name] = v

            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(new_state_dict)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))
    for i in os.listdir(args['video_path']):
            video_path = os.path.join(args['video_path'], i)
            print(video_path)
            process_video(video_path, model,args['num_queries'],step)
            print(f"{i} is done")


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()

    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
