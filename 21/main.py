#-*-coding:utf-8-*-

import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./components/") # 添加模型组件路径

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= " DpCas : << Deep Learning Componentized Application System >> ")
    parser.add_argument('-app', type=int, default = 0,
        help = "handpose_x:0, gesture:1 ") # 设置 App Example

    app_dict = {
        0:"handpose_x",
        1:"gesture"}

    args = parser.parse_args()# 解析添加参数

    APP_P = app_dict[args.app]

    # if APP_P == "handpose_x": # 手势识别
    from applications.handpose_local_app import main_handpose_x #加载 handpose 应用
    cfg_file = "./lib/hand_lib/cfg/handpose.cfg"
    main_handpose_x(cfg_file)#加载 handpose 应用



    print(" well done ~")
