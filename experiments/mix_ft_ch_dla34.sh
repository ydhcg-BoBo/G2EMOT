cd src
python train.py --exp_id mot17_half_og --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --gpus '0,1' --batch_size '12'
cd ..