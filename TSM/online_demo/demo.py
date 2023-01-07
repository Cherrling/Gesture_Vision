#coding=gbk
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from online_demo.mobilenet_v2_tsm import MobileNetV2
import cv2
import numpy as np
import torchvision
from PIL import Image
import time

SOFTMAX_THRES = 1
HISTORY_LOGIT = True
REFINE_OUTPUT = True

shift_buffer = [torch.zeros([1, 3, 56, 56]),
                torch.zeros([1, 4, 28, 28]),
                torch.zeros([1, 4, 28, 28]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 12, 14, 14]),
                torch.zeros([1, 12, 14, 14]),
                torch.zeros([1, 20, 7, 7]),
                torch.zeros([1, 20, 7, 7])]


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    # 等于这些类时默认为2
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    # 做其他事情默认也为2
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]  

    return history[-1], history  # 返回本帧结果和历史结果

catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


# catigories = [
#     "做其他事情",  # 0
#     "Drumming Fingers",  # 1
#     "没有手势",  # 2
#     "Pulling Hand In",  # 3
#     "把两根手指往里拉",  # 4
#     "手推掉",  # 5
#     "推开两根手指",  # 6
#     "向后滚动手",  # 7
#     "向前滚动手",  # 8
#     "颤抖的手",  # 9
#     "向下滑动两根手指",  # 10
#     "向左滑动两根手指",  # 11
#     "向右滑动两根手指",  # 12
#     "向上滑动两根手指",  # 13
#     "停止手势",  # 14
#     "刷下来",  # 15
#     "向左刷",  # 16
#     "向右刷",  # 17
#     "向上刷",  # 18
#     "拇指向下",  # 19
#     "拇指向上",  # 20
#     "顺时针",  # 21
#     "逆时针",  # 22
#     "全手放大",  # 23
#     "两根手指放大",  # 24
#     "全手缩小",  # 25
#     "Z两根手指缩小"  # 26
# ]

n_still_frame = 0

WINDOW_NAME = 'Video Gesture Recognition'

def main():

    torch_module = MobileNetV2(n_class=27)
    torch_module.load_state_dict(torch.load(r"C:\Code\Gesture_Vision\TSM\checkpoint\100.best.pth"))   # 加载模型，路径自己修改

    torch_module.eval()

    print("Open camera...")
    cap = cv2.VideoCapture(0)  # 打开摄像头
    
    cap.set(cv2.CAP_PROP_EXPOSURE, -8) # 设置相机曝光，（注意：不是所有相机有效）
    

    print(cap)

    # set a lower resolution for speed up   为加速设置一个较低的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables  窗口变量
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()  # 预处理
    print("Build Executor...")

    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = -1

    print("Ready!")
    while True:  # 读取摄像头
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate ， 隔帧抽取
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])  # 图片预处理
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))  # 张量转换
            with torch.no_grad():
                feat, *buffer = torch_module(input_var, *shift_buffer)

            if SOFTMAX_THRES > 0:
                feat_np = feat.numpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.numpy(), axis=1)[0]
                else:
                    idx_ = idx
                print(idx_)
            else:
                idx_ = np.argmax(feat.numpy(), axis=1)[0]  # 得到结果值

            if HISTORY_LOGIT:  # 平均
                history_logit.append(feat.numpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)  # 本帧结果、历史帧结果（最大保留20帧）

            t2 = time.time()
            print(f"{index} {catigories[idx]}")

            current_time = t2 - t1  # 推理时间

        # 识别效果展示部分
        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


main()

