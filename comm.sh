
# Train
nohup python ./train.py --dataset CUB  --model Conv4 --method relationnet --n_shot 1 --train_aug --gpu 0 > conv4_relati_cub_1shot.out 2>&1 &
nohup python ./train.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 1 --train_aug --gpu 0 > conv4_cosine_cub_1shot.out 2>&1 &
nohup python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0 > conv4_ournet_cub_1shot.out 2>&1 &

nohup python ./train.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0 > conv4_relati_cub_5shot.out 2>&1 &
nohup python ./train.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0 > conv4_cosine_cub_5shot.out 2>&1 &
nohup python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0 > conv4_ournet_cub_5shot.out 2>&1 &
 

# save features
python ./save_features.py --dataset CUB  --model Conv4 --method relationnet --n_shot 1 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 1 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0

python ./save_features.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0


# Test
python ./test.py --dataset CUB  --model Conv4 --method relationnet --n_shot 1 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 1 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 1 --train_aug --gpu 0

python ./test.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0





