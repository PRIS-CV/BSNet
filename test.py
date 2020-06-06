import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.relationnet import RelationNet
from methods.cosine_batch import CosineBatch
from methods.ournet import OurNet
from io_utils import parse_args, get_resume_file, get_best_file

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query

    if params.method in ['OurNet']:
        scores1, scores2 = model.set_forward(z_all, is_feature = True)
        scores = (scores1 + scores2)/2.
    else:
        scores  = model.set_forward(z_all, is_feature = True)

    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc

if __name__ == '__main__':
    params = parse_args('test')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
    feature_model = backbone.Conv4NP

    loss_type = 'mse'

    if params.method in ['relationnet']:
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['OurNet']:
        model = OurNet(feature_model, loss_type = loss_type, **few_shot_params)
    elif params.method in ['CosineBatch']:
        model = CosineBatch(feature_model, loss_type = loss_type, **few_shot_params)
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    print('checkpoint_dir:', checkpoint_dir)

    #modelfile   = get_resume_file(checkpoint_dir)


    modelfile = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split


    novel_file = os.path.join(checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query = 15, **few_shot_params)
        acc_all.append(acc)

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''

        exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
