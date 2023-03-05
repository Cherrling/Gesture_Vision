
import os
import time
from ops.models import TSN
from ops.transforms import *
import cv2
from PIL import Image
from numpy.random import randint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GroupScale_hyj(object):
    def __init__(self,input_size):
        self.input_size = input_size
        self.interpolation = Image.BILINEAR

    # @classmethod
    def _black_resize_img(self,ori_img):

        new_size = self.input_size
        ori_img.thumbnail((new_size,new_size))
        w2,h2 = ori_img.size
        bg_img = Image.new('RGB',(new_size,new_size),(0,0,0))
        if w2 == new_size:
            bg_img.paste(ori_img, (0, int((new_size - h2) / 2)))
        elif h2 == new_size:
            bg_img.paste(ori_img, (int((new_size - w2) / 2), 0))
        else:
            bg_img.paste(ori_img, (int((new_size - w2) / 2), (int((new_size - h2) / 2))))

        return bg_img

    def __call__(self,img_group):

        ret_img_group = [self._black_resize_img(img) for img in img_group]

        return ret_img_group

arch = 'resnet50'
num_class = 100
num_segments = 8
modality = 'RGB'
base_model = 'resnet50'
consensus_type='avg'
dataset = 'ucf101'
dropout = 0.8
img_feature_dim = 256
no_partialbn = True
pretrain = 'imagenet'
shift = True
shift_div = 8
shift_place = 'blockres'
temporal_pool = False
non_local = False
tune_from = None



video_path = r'C:\Code\Gesture_Vision\TSM\data\diqiushixingxing.mp4'
video_path = r'C:\Code\Gesture_Vision\TSM\data\tatongxueshijingcha.mp4'

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

model = torch.nn.DataParallel(model, device_ids=None).to(device)
# resume = r'C:\Code\Gesture_Vision\TSM\checkpoint\500_e50.tar'
resume = r'C:\Code\Gesture_Vision\TSM\checkpoint\100.best.pth'
checkpoint = torch.load(resume,map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
transform_hyj = torchvision.transforms.Compose([
    GroupScale(320),
    GroupCenterCrop(224),
    Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
    normalize,
])



state=-1

# 存放切分后的视频
pil_img_list = list()
cap = cv2.VideoCapture(video_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

average_duration = frame_count // num_segments
if average_duration > 0:
    offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration,size=num_segments)
    for off_item in offsets:
        cap.set(cv2.CAP_PROP_POS_FRAMES,off_item)
        ret, frame = cap.read()
        if ret:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_img_list.extend([frame_pil])

input = transform_hyj(pil_img_list)
# input = input.unsqueeze(0).cuda()
input = input.unsqueeze(0).to(device)
out = model(input)
output_index = int(torch.argmax(out).cpu())
state = output_index
print("out", output_index)


# ----------
exit()

cap = cv2.VideoCapture(video_path)
start_time = time.time()
counter = 0
frame_numbers = 0
fps = cap.get(cv2.CAP_PROP_FPS)
if fps < 1:
    fps = 30

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_numbers+=1
        print(frame_numbers)
        key = cv2.waitKey(1) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break
        counter += 1
        if (time.time() - start_time) != 0:
            cv2.putText(frame, "{0} {1}".format((state),float('%.1f' % (counter / (time.time() - start_time)))), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
            cv2.imshow('frame', frame)

            counter = 0
            start_time = time.time()
        time.sleep(1 / fps)

    else:
        break

cap.release()
cv2.destroyAllWindows()
