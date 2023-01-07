python main.py ucf101 RGB --arch resnet50 --num_segment 2 --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 6 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth

python main.py ucf101 RGB --arch mobilenetv2 --num_segment 8 --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth


python main.py ucf101 RGB --arch resnet --num_segment 8 --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
