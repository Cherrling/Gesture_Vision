#coding=gbk
import os
import time
from ops.models import TSN
from ops.transforms import *
import cv2
from PIL import Image

# arch = 'mobilenetv2'
arch = 'resnet50'
num_class = 2
num_segments = 8
modality = 'RGB'
# base_model = 'mobilenetv2'
base_model = 'resnet50'
consensus_type='avg'
dataset = 'ucf101'
dropout = 0.1
img_feature_dim = 256
no_partialbn = True
pretrain = 'imagenet'
shift = True
shift_div = 8
shift_place = 'blockres'
temporal_pool = False
non_local = False
tune_from = None


#load model
model = TSN(num_class, num_segments, modality,
                base_model=arch,
                consensus_type=consensus_type,
                dropout=dropout,
                img_feature_dim=img_feature_dim,
                partial_bn=not no_partialbn,
                pretrain=pretrain,
                is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                fc_lr5=not (tune_from and dataset in tune_from),
                temporal_pool=temporal_pool,
                non_local=non_local)

model = torch.nn.DataParallel(model, device_ids=None).cuda()
resume = 'C:/Code/Gesture_Vision/TSM/checkpoint/100.best.pth' #  the last weights
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])
checkpoint.pop('linear.weight')
checkpoint.pop('linear.bias')
model.eval()

#how to deal with the pictures
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
transform_hyj = torchvision.transforms.Compose([
    GroupScale_hyj(input_size=320),
    Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
    normalize,
])

video_path = 'C:\Code\Gesture_Vision\TSM\data/1.avi'

pil_img_list = list()

cls_text = ['nofight','fight']
cls_color = [(0,255,0),(0,0,255)]

import time

cap = cv2.VideoCapture(video_path) #导入的视频所在路径
start_time = time.time()
counter = 0
frame_numbers = 0
training_fps = 30
training_time = 2.5
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
if fps < 1:
    fps = 30
duaring = int(fps * training_time / num_segments)
print(duaring)
# exit()


state = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_numbers+=1
        print(frame_numbers)
        # print(len(pil_img_list))
        if frame_numbers%duaring == 0 and len(pil_img_list)<8:
            frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            pil_img_list.extend([frame_pil])
        if frame_numbers%duaring == 0 and  len(pil_img_list)==8:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img_list.pop(0)
            pil_img_list.extend([frame_pil])
            input = transform_hyj(pil_img_list)
            input = input.unsqueeze(0).cuda()
            out = model(input)
            print(out)
            output_index = int(torch.argmax(out).cpu())
            state = output_index

        #键盘输入空格暂停，输入q退出
        key = cv2.waitKey(1) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break
        counter += 1#计算帧数
        if (time.time() - start_time) != 0:#实时显示帧数
            cv2.putText(frame, "{0} {1}".format((cls_text[state]),float('%.1f' % (counter / (time.time() - start_time)))), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, cls_color[state],3)
            cv2.imshow('frame', frame)

            counter = 0
            start_time = time.time()
        time.sleep(1 / fps)#按原帧率播放
        # time.sleep(2/fps)# observe the output
    else:
        break

cap.release()
cv2.destroyAllWindows()

