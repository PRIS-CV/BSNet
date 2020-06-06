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

* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

## Training

* method: relationnet|CosineBatch|OurNet.
* n_shot: number of labeled data in each class （1|5）.
* train_aug: perform data augmentation or not during training.
* gpu: gpu id.

```shell
python ./train.py --dataset CUB  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./train.py --dataset CUB  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./train.py --dataset CUB  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
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


## Citation

If you find this paper useful in your research, please consider citing:

```

```

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework, Backbone, Dataset, Method: Relational Network
  https://github.com/wyharveychen/CloserLookFewShot 

  

