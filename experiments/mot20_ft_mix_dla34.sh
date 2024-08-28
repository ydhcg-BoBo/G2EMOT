cd src
python train.py --gpus 0,1 --batch_size 12 --exp_id mot20_ftmot20_4 --load_model '../models/crowdhuman_exp12.pth' --num_epochs 20 --lr_step '10' --data_cfg '../src/lib/cfg/mot20.json'
cd ..