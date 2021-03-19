import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.backbone import Resnet
from config import settings

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.backbone = Resnet.resnet18(pretrained=True)

        self.bn = nn.BatchNorm1d(64)
        self.session_len = settings.CUB200_SessLen

        self.node = 512
        self.fc1 = nn.Linear(512, self.node, bias=False) #
        self.fc2 = nn.Linear(self.node, self.session_len[0], bias=False)
        SessLen = len(self.session_len)
        for i in range(1, SessLen):
            exec('self.fc'+str(i+2)+ '= nn.Linear(self.node, self.session_len[i], bias=False)')
        for i in range(1, SessLen):
            exec ('self.fc_aux' + str(i + 2) + '= nn.Linear(512, self.node, bias=False)')
        
        Alpha = torch.zeros(SessLen, self.node)
        Alpha[0] = Alpha[0] + 1
        self.register_buffer('Alpha', Alpha)
        self.r = nn.Parameter(torch.tensor(0.0))
        self.gamma = args.gamma
        self.Gamma = [1]*SessLen
        self.bce_logits_func = nn.CrossEntropyLoss()
        self.sess = 0

    def forward(self, x, sess=0, epoch=0, Mode='train'):
        self.sess = sess
        if sess>0:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        out1 = self.fc1(x.view(x.size(0), -1))
        out = self._l2norm(out1, dim=1)
        for i in range(sess + 1):
            if i == 0:
                output = self.fc2(out)
            else:
                fc = eval('self.fc' + str(i + 2))
                fc_aux = eval('self.fc_aux' + str(i + 2))
                out_aux = fc_aux(x.view(x.size(0), -1))

                if i< sess:
                    out_aux = out_aux*self.Alpha[i]
                else:
                    if Mode=='train':
                        beta = 1.0+max(epoch-20,0)
                        t = torch.mean(out_aux,dim=0) 
                        self.alpha = torch.sigmoid(beta*t)
                        out_aux = out_aux*self.alpha
                    else:
                        out_aux = out_aux*self.Alpha[i]
                new_node = out1*self.gamma+out_aux
                new_node = self._l2norm(new_node, dim=1)
                output = torch.cat([output, fc(new_node)], dim=1)  # +out_aux

        return out, output

    def finish_train(self):
        self.Alpha[self.sess] = self.alpha.detach() #data.
        #pass

    def _l2norm(self, inp, dim=1):
        '''Normlize the inp tensor with l2-norm.'''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def get_loss(self, pred, label, output_old=None, logits=None):
        loss_bce_seg = self.bce_logits_func(pred, label.long())
        loss_dis = 0
        loss_cos = 0
        R1=0
        if output_old is not None:
            loss_dis = self.distillation_loss(pred, output_old)
        if logits is not None:
            out, out_aux = logits
            cos_loss = nn.CosineEmbeddingLoss()
            loss_cos = cos_loss(out, out_aux, torch.tensor(-1.0).cuda())  #
        if self.sess>0:
            R1 = torch.sum(nn.ReLU()(torch.norm(self.alpha, p=1, dim=0)/self.node - self.r))
        return loss_bce_seg + loss_dis + 0.1*R1

    def distillation_loss(self, pred_N, pred_O, T=2):
        if pred_N.shape[1] != pred_O.shape[1]:
            pred_N = pred_N[:, :pred_O.shape[1]]
        outputs = torch.log_softmax(pred_N / T, dim=1)  # compute the log of softmax values
        labels = torch.softmax(pred_O / T, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        loss = -torch.mean(outputs, dim=0, keepdim=False)

        return loss


