# BSNet
Code release for the paper [BSNet: Bi-Similarity Network for Few-shot Fine-grained Image Classification](#).

## Requirements

* python=3.6
* PyTorch=1.2+
* torchvision=0.4.2
* pillow=6.2.1
* numpy=1.18.1
* h5py=1.10.2

## Dataset

#### CUB-200-2011

## Training

* dataset: CUB/dogs/cars/aircraft.
* model: the type of embedding module. Conv{4|6} / ResNet{10|18|34|50|101}.
* method: relationnet/CosineBatch/OurNet.
* n_shot: number of labeled data in each class, same as n_support.
* train_aug: perform data augmentation or not during training.
* gpu: gpu id.

```shell
python ./train.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./train.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 1
python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 2
```

## Save features

```shell
python ./save_features.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```

## Test

```shell
python ./test.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```

## results

<table>
    <tr>
        <td colspan="3" align='center'>CUB-200-2011</td>
    </tr>
    <tr>
        <td align='center'></td>
        <td align='center'>5-shot</td>
        <td align='center'>1-shot</td>
    </tr>
    <tr>
        <td align='center'>RelationNet</td>
        <td align='center'></td>
        <td align='center'></td>
    </tr>
    <tr>
        <td align='center'>CosineNet</td>
        <td align='center'></td>
        <td align='center'></td>
    </tr>
    <tr>
        <td align='center'>R&C (Ours)</td>
        <td align='center'></td>
        <td align='center'></td>
    </tr>
</table>






