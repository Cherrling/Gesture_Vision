

# 引入包
import os
import glob
import fnmatch
# 视频分帧后的根目录
img_root = "TSM\data\img"
#放入训练集中的视频个数，其余放入测试集
num=200
# 标注文件根目录
label_root = "TSM\data\label"

# # 获取对应的label
# def get_label():
#     res = []
#     with open(label_root + 'corpus.txt', encoding='utf-8') as f:
#         temp = f.read().split("\n")
#         for item in temp:
#             res.append(item.split(" "))
#     return res

def parse_directory(root,rgb_prefix='image_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    frame_folders = []
    frame = glob.glob(os.path.join(root, "*"))
    for frame_name in frame:
        frame_path = glob.glob(os.path.join(frame_name, '*'))
        frame_folders.extend(frame_path)

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    dir_dict = {}
    rgb_counts = {}
    flow_counts = {}    
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('\\')[-1]
        rgb_counts[k] = all_cnt[0]
        dir_dict[k] = f

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: ' + f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

    print('frame folder analysis done')


    return dir_dict, rgb_counts, flow_counts


# 给出生成的目录
# 抽帧后dir,num
def build_list(f_info,root):
    # for item in f_info[1]:
    #     print(item)
    train_list = []
    val_list = []
    i=0
    for key in f_info[0]:
        temp_dir =root + '/' + f_info[0][key].split('\\')[-2] + '/' + f_info[0][key].split('\\')[-1]
        temp_num = f_info[1][key]
        temp_cla = int(f_info[0][key].split('\\')[-2])
        if i < num:
            train_list.append('{} {} {}\n'.format(temp_dir, temp_num, temp_cla))
        else:
            val_list.append('{} {} {}\n'.format(temp_dir, temp_num, temp_cla))
        i+=1
        if i >= 250:
            i=0

    # for i in range(0,len(f_info[0])):
    #     for j in range(0,len(f_info[0][i])):
    #         temp_dir =root + '/' + f_info[0][i][j].split('\\')[-2] + '/' + f_info[0][i][j].split('\\')[-1]
    #         temp_num = f_info[1][i][j]
    #         temp_cla = int(f_info[0][i][j].split('\\')[-2])
    #         if j<200:
    #             train_list.append('{} {} {}\n'.format(temp_dir, temp_num, temp_cla))
    #         else:
    #             val_list.append('{} {} {}\n'.format(temp_dir, temp_num, temp_cla))
            



    return train_list,val_list

if __name__ == '__main__':
    f_info = parse_directory(img_root)
    for_list = build_list(f_info,img_root)
    
    with open(label_root+'train.txt','w') as f:
        f.writelines(for_list[0])
        
    with open(label_root+'val.txt','w') as f:
        f.writelines(for_list[1])