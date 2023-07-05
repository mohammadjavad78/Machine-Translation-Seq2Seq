import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from dataloaders.Splitter import *

import pickle


class MyDataset(Dataset):
    def __init__(self,train=True):
        super(MyDataset, self).__init__()
        self.embedding_matrix=None
        spli=Splitter()
        if(train):
            self.Eng_Spa,self.MAX_ENG,self.MAX_SPA=spli.get_train()
        else:
            self.Eng_Spa,self.MAX_ENG,self.MAX_SPA=spli.get_test()

        with open('./datasets/Eng_dict.pkl', 'rb') as f:
            self.Eng_array = pickle.load(f)

        with open('./datasets/Spa_dict.pkl', 'rb') as f:
            self.Spa_array = pickle.load(f)




    def __getitem__(self, index):
        iloc=self.Eng_Spa.iloc[index]
        Eng_tokenized=Tokenization(iloc['Eng'])
        Spa_tokenized=Tokenization(iloc['Spa'])
        
    
                
        Eng=np.ones(self.MAX_ENG+2)
        for i in range(len(Eng_tokenized)):
            Eng[i]=self.Eng_array.get(Eng_tokenized[i],2)


        Spa=np.zeros((self.MAX_SPA+2,len(self.Spa_array)+1))
        Spa[:,1]=1
        for i in range(len(Spa_tokenized)):
            Spa[i][1]=0
            Spa[i][self.Spa_array.get(Spa_tokenized[i],2)]=1


            
        Spa2=np.ones(self.MAX_SPA+2)
        for i in range(len(Spa_tokenized)-1):
            Spa2[i]=self.Spa_array.get(Spa_tokenized[i+1],2)


        return torch.IntTensor(Eng),torch.FloatTensor(Spa),torch.IntTensor(Spa2)
        

    def __len__(self):
        return len(self.Eng_Spa)
    

    def get_embedding_matrix(self):
        return self.embedding_matrix
    
    def get_MAX_ENG(self):
        return self.MAX_ENG

    def get_MAX_SPA(self):
        return len(self.Spa_array)+1
    
    def get_MAX_SPA_perline(self):
        return self.MAX_SPA
        
    