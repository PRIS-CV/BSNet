# BSNet
Code release for the paper [BSNet: Bi-Similarity Network for Few-shot Fine-grained Image Classification](#). (TIP2020)

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

## Train

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
        <td align='center'>5-way 5-shot Accuracy (%)</td>
        <td align='center'>5-way 1-shot Accuracy (%)</td>
    </tr>
    <tr>
        <td align='center'>Relation Network</td>
        <td align='center'>77.87 &plusmn; 0.64</td>
        <td align='center'>63.94 &plusmn; 0.92</td>
    </tr>
    <tr>
        <td align='center'>Cosine Network</td>
        <td align='center'>77.86 &plusmn; 0.68</td>
        <td align='center'>65.04 &plusmn; 0.97</td>
    </tr>
    <tr>
        <td align='center'>BSNet (R&C)</td>
        <td align='center'><b>80.99 &plusmn; 0.63</b></td>
        <td align='center'><b>65.89 &plusmn; 1.00</b></td>
    </tr>
</table>



## Citation

If you find this paper useful in your research, please consider citing:

```

```

## References
Our code is based on Chen's contribution. Specifically, except for our core design, cosine network and BSNet, everything else （e.g. backbone, dataset, relation network, evaluation standards, hyper-parameters）are built on and integrated in https://github.com/wyharveychen/CloserLookFewShot.
  

