
import torch
import torch.nn as nn
from nets.Encoder import *
from nets.Decoder import *
import pandas as pd
from dataloaders.DataLoader import *


class Seq2SeqTrans(nn.Module):
    def __init__(self,numlayer=3,teacherforcing=0.5,dropout=0):
        super(Seq2SeqTrans,self).__init__()
        embedding_matrix=pd.read_csv('./datasets/MatrixEmbedded.csv',names=[i for i in range(100)])
        embedding_matrix = torch.FloatTensor(embedding_matrix.values)
        train_dataset=MyDataset(train=True)
        numlayer=3
        teacherforcing=0.5
        self.ER=Encoder(embedding_matrix,numlayer,dropout=dropout)
        self.DE=Decoder(embedding_matrix.shape[1],train_dataset.get_MAX_SPA(),train_dataset.get_MAX_SPA_perline()+2,teacherforcing,numlayer,dropout=dropout)

    def forward(self,input,target):
        output,hidden1=self.ER(input)
        output,hidden1=self.DE(target,hidden1)
        return output
    
ss=Seq2SeqTrans()


