# This code is modified from https://github.com/wyharveychen/CloserLookFewShot

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils




from torch.nn import functional as F


class Classifier_cosine(nn.Module):
    def __init__(self, n_way, n_query, n_support, feat_dim):
        super(Classifier_cosine, self).__init__()
        self.n_way = n_way
        self.feat_dim = feat_dim

        self.n_support = n_support

        self.layer1 = nn.Sequential(
                        nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3,padding=1),
                        nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feat_dim[0],feat_dim[0],kernel_size=3,padding=1),
                        nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True),
                        nn.ReLU(),
                        nn.AvgPool2d(2)
                        )

    def forward(self, z_support, z_query):

        self.n_query = z_query.size(1) # 16
        extend_final_feat_dim = self.feat_dim.copy() # [64, 19, 19]

        z_support = z_support.contiguous().view( self.n_way, self.n_support, *self.feat_dim ).mean(1)       # [5, 64, 19, 19]
        z_query = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )         # [80, 64, 19, 19] 

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)         # [80, 5, 64, 19, 19] 
        z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)      # [5, 80, 64, 19, 19]
        z_query_ext = torch.transpose(z_query_ext,0,1)      # [80, 5, 64, 19, 19]
        z_support_ext = z_support_ext.view(-1, *extend_final_feat_dim)      # [400, 64, 19, 19]
        z_query_ext = z_query_ext.contiguous().view(-1, *extend_final_feat_dim)         # [400, 64, 19, 19]

        x_support = self.layer1(z_support_ext)      # [400, 64, 19, 19] ==> [400, 64, 9, 9]
        x_query = self.layer1(z_query_ext)      # [400, 64, 19, 19] ==> [400, 64, 9, 9]

        x_support = self.layer2(x_support)      # [400, 64, 9, 9] ==> [400, 64, 4, 4]
        x_query = self.layer2(x_query)      # [400, 64, 9, 9] ==> [400, 64, 4, 4]

        
        x_support_flat = x_support.view(self.n_way*self.n_way*self.n_query, -1)         # [400, 1024]
        x_query_flat = x_query.view(self.n_way*self.n_way*self.n_query, -1)         # [400, 1024]


        # computer the cosine distance between x_support and x_query [400]
        cosine = F.cosine_similarity(x_support_flat, x_query_flat, dim=1).view(self.n_way*self.n_way*self.n_query)


        return cosine


class CosineBatch(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = 'mse'):
        super(CosineBatch, self).__init__(model_func,  n_way, n_support)

        self.loss_type = loss_type  #'softmax'# 'mse'

        self.classifier_cosine = Classifier_cosine(n_way=self.n_way, n_query=self.n_query, n_support=self.n_support, feat_dim=self.feat_dim)


        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        cosine = self.classifier_cosine(z_support, z_query).view(-1, self.n_way)

        return cosine

    
    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

        scores = self.set_forward(x)
        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = Variable(y_oh.cuda())            

            return self.loss_fn(scores, y_oh )
        else:
            y = Variable(y.cuda())
            return self.loss_fn(scores, y )