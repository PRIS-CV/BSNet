import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.relationnet import RelationNet
from methods.cosine_batch import CosineBatch
from methods.ournet import OurNet
from io_utils import parse_args, get_resume_file  

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)


    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    image_size = 84

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 600
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600 #default
     



    if params.method in ['relationnet', 'CosineBatch', 'OurNet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        


        feature_model = backbone.Conv4NP

        loss_type = 'mse'
        if params.method == 'relationnet':
            print('method:', params.method)
            model           = RelationNet(feature_model, loss_type = loss_type, **train_few_shot_params)
        elif params.method == 'CosineBatch':
            print('method:', params.method)
            model = CosineBatch(feature_model, loss_type = loss_type, **train_few_shot_params)
        elif params.method == 'OurNet':
            print('method:', params.method)
            model = OurNet(feature_model, loss_type = loss_type, **train_few_shot_params)

    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
