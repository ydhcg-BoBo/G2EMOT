cd src
python train.py --exp_id crowdhuman_exp12_p4_e6 --gpus 2,3 --batch_size 6 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..