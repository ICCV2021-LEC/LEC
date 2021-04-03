import os
import json
import argparse
import time
import numpy as np
import copy

import torch
import torch.optim as optim
from config import settings
from data.LoadData import data_loader
from data.LoadData import val_loader
from models import *
# from utils import Log
from utils import Restore

decay_epoch = [1000]
decay=0.5

def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int,default=None,help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--start_sess", type=int,default='1')
    parser.add_argument("--max_epoch", type=int,default='1000')
    parser.add_argument("--batch_size", type=int,default='128')
    parser.add_argument("--dataset", type=str,default='CUB200')
    parser.add_argument("--arch", type=str,default='LECNet') #quickcnn_v4 ExpNetQ_v1_1
    parser.add_argument("--lr", type=float,default=0.005)#0.005 0.002
    parser.add_argument("--r", type=float,default=0.1)#0.01
    parser.add_argument("--gamma", type=float,default=0.6)#0.01
    parser.add_argument("--lamda", type=float,default=1.0)#0.01
    parser.add_argument("--seed", type=str,default='Seed_3')#0.01 #Seed_1
    #parser.add_argument("--decay_epoch", nargs='+', type=int, default=[50])

    return parser.parse_args()


def test(args, network):
    TP = 0.0
    All = 0.0
    val_data = val_loader(args)
    network.eval()
    for i, data in enumerate(val_data):
        img, label = data
        img, label = img.cuda(), label.cuda()       
        out, output = network(img, sess=args.sess, Mode='test')
        #out, output = network(img, args.sess)
        _, pred = torch.max(output, dim=1)
        TP += torch.eq(pred, label).sum().float().item()
        All += torch.eq(label, label).sum().float().item()
    
    acc = float(TP)/All
    network.train()
    return acc

def test_continue(args, network):
    val_data = val_loader(args)
    acc_list = []
    network.eval()
    for sess in range(args.sess+1):
        TP = 0.0
        All = 0.0
        val_data.dataset.Update_Session(sess)
        for i, data in enumerate(val_data):
            img, label = data
            img, label = img.cuda(), label.cuda()       
            out, output = network(img, args.sess, Mode='test')
            _, pred = torch.max(output, dim=1)
            
            
            TP += torch.eq(pred, label).sum().float().item()
            All += torch.eq(label, label).sum().float().item()
    
        acc = float(TP)/All
        acc_list.append(acc)
    network.train()
    return acc_list

def acc_list2string(acc_list):
    acc_str=''
    for idx,item in enumerate(acc_list):
        acc_str +='Sess%d: %.3f, '%(idx, item)
    
    return acc_str

def Trans_ACC(args, acc_list):
    if args.dataset=='CUB200':
        SessLen = settings.CUB200_SessLen
    if args.dataset=='CIFAR100':
        SessLen = settings.CIFAR100_SessLen
    if args.dataset=='miniImageNet':
        SessLen = settings.miniImagenet_SessLen
    ACC = 0
    ACC_A = 0
    ACC_M=0
    num = 0
    for idx, acc in enumerate(acc_list):
        ACC+=acc*SessLen[idx]
        num+=SessLen[idx]
        if idx ==args.sess:
            ACC_A+=acc
        else:
            ACC_M+=acc*SessLen[idx]
    ACC=ACC/num
    ACC_M=ACC_M/(num-SessLen[idx])
    return ACC, ACC_A, ACC_M

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def train(args):
    ACC_list = []
    ACC_list_train=[]
    lr = args.lr
    network = eval(args.arch).OneModel(args) #fc:fc1  fw:sess-1 fc
    network.cuda()
    print(network)
    if args.start_sess>0:
        Restore.load(args, network, filename='Sess0.pth.tar')
        args.sess = args.start_sess-1
        ACC = test(args, network)
        ACC_list.append(ACC)
        print('Sess: %d'%args.sess, 'acc_val: %f'%ACC)
    for sess in range(args.start_sess, args.sesses+1):
        Restore.load(args, network, filename='Sess'+str(sess)+'.pth.tar')
        args.sess = sess
        Best_ACC = 0  
        loss_list = []
        begin_time = time.time()
        ACC_Sess = test_continue(args, network)
        ACC_Sess_str = acc_list2string(ACC_Sess)
        ACC, ACC_A, ACC_M = Trans_ACC(args, ACC_Sess)
        print('Sess:%d: ACC:%.4f ACC_A:%.4f ACC_M:%.4f'%(sess, ACC, ACC_A, ACC_M))
        ACC_list.append(ACC)
    timestamp = time.strftime("%m%d-%H%M", time.localtime())
    print('ACC:', ACC_list)
    print('End')


if __name__ == '__main__':
    args = get_arguments()
    if args.sesses==None:
        if args.dataset=='CUB200':
            SessLen = len(settings.CUB200_SessLen)
        if args.dataset=='CIFAR100':
            SessLen = len(settings.CIFAR100_SessLen)
        if args.dataset=='miniImageNet':
            SessLen = len(settings.miniImagenet_SessLen)
        args.sesses = SessLen-1
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
