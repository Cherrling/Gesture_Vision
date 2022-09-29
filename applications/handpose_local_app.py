#-*-coding:utf-8-*-

import cv2
import time

from multiprocessing import Process
from multiprocessing import Manager

import cv2
# import numpy as np
# import random
import time

# 加载模型组件库
from hand_detect.yolo_v3_hand import yolo_v3_hand_model
from hand_keypoints.handpose_x import handpose_x_model
# from classify_imagenet.imagenet_c import classify_imagenet_model

# 加载工具库
import sys
sys.path.append("./lib/hand_lib/")
from cores.handpose_fuction import handpose_track_keypoints21_pipeline
# from cores.handpose_fuction import hand_tracking,audio_recognize,judge_click_stabel,draw_click_lines
from cores.handpose_fuction import hand_tracking
from utils.utils import parse_data_cfg

'''
/*****************************************/
                算法 pipeline
/*****************************************/
'''


def handpose_x_process(info_dict,config):
    # 模型初始化
    print("load model component  ...")
    # yolo v3 手部检测模型初始化
    hand_detect_model = yolo_v3_hand_model(conf_thres=float(config["detect_conf_thres"]),nms_thres=float(config["detect_nms_thres"]),
        model_arch = config["detect_model_arch"],model_path = config["detect_model_path"],yolo_anchor_scale = float(config["yolo_anchor_scale"]),
        img_size = float(config["detect_input_size"]),
        )
    # handpose_x 21 关键点回归模型初始化
    handpose_model = handpose_x_model(model_arch = config["handpose_x_model_arch"],model_path = config["handpose_x_model_path"])
    #
    gesture_model = None # 目前缺省

    cap = cv2.VideoCapture(int(config["camera_id"])) # 开启摄像机

    cap.set(cv2.CAP_PROP_EXPOSURE, -3) # 设置相机曝光，（注意：不是所有相机有效）

    # url="http://admin:admin@192.168.43.1:8081"
    # cap=cv2.VideoCapture(url)
    print("start handpose process ~")

    info_dict["handpose_procss_ready"] = True #多进程间的开始同步信号

    gesture_lines_dict = {} # 点击使能时的轨迹点

    hands_dict = {} # 手的信息
    hands_click_dict = {} #手的按键信息计数
    track_index = 0 # 跟踪的全局索引

    while True:
        ret, img = cap.read()# 读取相机图像
        if ret:# 读取相机图像成功
            # img = cv2.flip(img,-1)


            algo_img = img.copy()
            st_ = time.time()
            #------
            hand_bbox =hand_detect_model.predict(img,vis = True) # 检测手，获取手的边界框

            hands_dict,track_index = hand_tracking(data = hand_bbox,hands_dict = hands_dict,track_index = track_index) # 手跟踪，目前通过IOU方式进行目标跟踪
            # 检测每个手的关键点及相关信息
            handpose_list = handpose_track_keypoints21_pipeline(img,hands_dict = hands_dict,hands_click_dict = hands_click_dict,track_index = track_index,algo_img = algo_img,
                handpose_model = handpose_model,gesture_model = gesture_model,
                icon = None,vis = True)
            et_ = time.time()

            #fps打印
            fps_ = 1./(et_-st_+1e-8)
            fps = "%.2f fps"%fps_
            # print(fps_)

            #------------------------------------------ 跟踪手的 信息维护
            #------------------ 获取跟踪到的手ID
            id_list = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_list.append(dict_["id"])
            # print(id_list)
            #----------------- 获取需要删除的手ID
            id_del_list = []
            for k_ in gesture_lines_dict.keys():
                if k_ not in id_list:#去除过往已经跟踪失败的目标手的相关轨迹
                    id_del_list.append(k_)
            #----------------- 删除无法跟踪到的手的相关信息
            for k_ in id_del_list:
                del gesture_lines_dict[k_]
                del hands_click_dict[k_]



            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0),5)
            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))
            cv2.putText(img, fps, (240,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

            print(handpose_list)

            cv2.namedWindow("image",0)
            cv2.imshow("image",img)
            if cv2.waitKey(1) == 27:
                info_dict["break"] = True
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main_handpose_x(cfg_file):
    config = parse_data_cfg(cfg_file)

    print("\n/---------------------- main_handpose_x config ------------------------/\n")
    for k_ in config.keys():
        print("{} : {}".format(k_,config[k_]))
    print("\n/------------------------------------------------------------------------/\n")

    print(" loading handpose_x local demo ...")
    g_info_dict = Manager().dict()# 多进程共享字典初始化：用于多进程间的 key：value 操作
    g_info_dict["handpose_procss_ready"] = False # 进程间的开启同步信号
    g_info_dict["break"] = False # 进程间的退出同步信号
    g_info_dict["double_en_pts"] = False # 双手选中动作使能信号

    g_info_dict["click_up_cnt"] = 0
    g_info_dict["click_dw_cnt"] = 0

    g_info_dict["reco_msg"] = None

    print(" multiprocessing dict key:\n")
    for key_ in g_info_dict.keys():
        print( " -> ",key_)
    print()

    #-------------------------------------------------- 初始化各进程
    process_list = []
    t = Process(target=handpose_x_process,args=(g_info_dict,config,))
    process_list.append(t)

    for i in range(len(process_list)):
        process_list[i].start()

    for i in range(len(process_list)):
        process_list[i].join()# 设置主线程等待子线程结束

    del process_list
