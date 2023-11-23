import argparse
import os
import sys
from pathlib import Path
import cv2
import time
from tqdm import tqdm
import pymysql
import datetime
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# 初始化相对路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

@torch.no_grad()
def person_count(
    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    '''
        建筑各区域内人员数量统计算法

        Args:
            weights：模型权重路径
            source：带处理图像路径
            data：配置文件路径
            imgsz：图像大小
            ...
        Return:
            area_total_person：各区域人员分布数量
            real_area：各区域名称
    '''
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    total_person = []
    path_ = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        person_num = 0
        path_.append(path)
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if names[int(c)] == 'person':
                        person_num = n.item()
                        break
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        total_person.append(person_num)
    area_total_person = total_person[7:]
    area_total_person.append(sum(total_person[:7]))
    real_area = ['F0-走廊1', 'F0-走廊2', 'F1-东走廊', 'F1-展厅1', 'F1-展厅2', 'F1-展厅3', 'F1-监控室', 'F1-调度室', 'F1-门斗', 'F2-东走廊', 'F2-中走廊', 'F2-会议室', 'F2-变电所1', 'F2-变电所2', 'F2-机房', 'F2-西走廊', 'F0-设备']
    print(area_total_person)
    print(real_area)

    return area_total_person, real_area

def person_predict(history_t_person_num, real_area):
    '''
        建筑各区域内人员数量预测算法

        Args:
            history_t_person_num：历史一段时间内各区域人员分布数量
            real_area：各区域名称
        Return:
            area_predict_person：各区域人员预测数量
    '''
    diff_period = []
    for item in history_t_person_num[:-1]:
        diff = np.subtract(history_t_person_num[-1], item)
        diff_period.append(diff)
    diff_period = np.stack(diff_period, axis=0)
    print(real_area)
    area_predict_person = []
    for i in range(diff_period.shape[1]):
        if np.sum(diff_period[:, i]) == 0:
            area_predict_person.append(history_t_person_num[-1][i])
        else:
            area_predict_person.append(history_t_person_num[-1][i] - round(np.sum(diff_period[:, i]) / diff_period.shape[0]))

    return area_predict_person
def write2db(area_total_person, real_area, area_predict_person):
    '''
        将人员分布与预测算法计算结果写入数据库对应的表中

        Args:
            area_total_person：各区域人员分布数量
            real_area：各区域名称
            area_predict_person：各区域人员预测数量
        Return:
            无
    '''
    # update 将计算结果写入数据库中
    dbname = 'yulin_esms'
    db = pymysql.connect(host='192.168.3.13', port=3306, user='root', password='123456', db=dbname)
    cursor = db.cursor()
    for name, num, pred_num in zip(real_area, area_total_person, area_predict_person):
        sql = "update rt_person_distribution set people_nums=%s where fig_name=%s"
        value = [name, str(num), str(pred_num)]
        cursor.execute(sql, value)
        db.commit()
    cursor.close()
    db.close()

def parse_opt():
    '''
    初始化模型参数

    Args:
        无
    Return:
        opt: 模型参数，例如模型权重文件路径、数据文件路径等
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    '''
        主函数：负责单次全区域摄像头的图像数据读取、保存以及运算

        Args:
            opt: 模型参数，例如模型权重文件路径、数据文件路径等
        Return:
            无
    '''
    # ip to area 点位表初始化
    ip2area = {
        '192.168.11.21': 'F1-展厅2',
        '192.168.11.22': 'F1-展厅3',
        '192.168.11.33': 'F1-展厅1',
        '192.168.11.28': 'F1-门斗',
        '192.168.11.38': 'F1-调度室',
        '192.168.11.31': 'F1-监控室',
        '192.168.11.37': 'F1-东走廊',
        '192.168.11.36': 'F2-变电所2',
        '192.168.11.29': 'F2-变电所1',
        '192.168.11.20': 'F2-东走廊',
        '192.168.11.35': 'F2-中走廊',
        '192.168.11.75': 'F2-西走廊',
        '192.168.11.23': 'F2-机房',
        '192.168.11.76': 'F2-会议室',
        '192.168.11.63': 'F0-走廊1',
        '192.168.11.59': 'F0-走廊2',
        '192.168.11.67': 'F0-设备1',
        '192.168.11.68': 'F0-设备2',
        '192.168.11.71': 'F0-设备3',
        '192.168.11.60': 'F0-设备4',
        '192.168.11.62': 'F0-设备5',
        '192.168.11.70': 'F0-设备6',
        '192.168.11.69': 'F0-设备7'
    }

    # 读取每个摄像头视频流，并保存图片到指定目录下
    for key, value in ip2area.items():
        url = 'rtsp://admin:Admin123@' + key
        cap = cv2.VideoCapture(url)
        while True:
            ret, frame = cap.read()
            if ret:
                break
        cv2.imwrite('./image/' + value + '.png', frame)
        cap.release()

    # 运行目标检测算法，统计各个区域内人数
    check_requirements(exclude=('tensorboard', 'thop'))
    area_total_person, real_area = person_count(**vars(opt))
    # 运行人流预测算法，预测各个区域内人数
    demo = np.array([[0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,0,0,3,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,0,0,3,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,0,0,3,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,0,0,3,0,0,0,0,0,0,0,0,0]])
    area_predict_person = person_predict(history_t_person_num=demo, real_area=real_area)
    # 将结果写入数据库中
    write2db(area_total_person, real_area, area_predict_person)


if __name__ == "__main__":
    opt = parse_opt()
    n = 0
    # 每隔五分钟运行一次
    while True:
        print(n)
        date = datetime.datetime.now()
        print(date)
        main(opt)

        s = time.time()
        time.sleep(300)
        print('等待时间为：(分钟)', (time.time() - s) / 60)
        n += 1

