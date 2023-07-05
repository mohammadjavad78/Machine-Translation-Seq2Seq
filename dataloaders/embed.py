import pandas as pd
import csv
from Splitter import *
import pickle 

def EmbedEng():
    spli=Splitter()
    glove=pd.read_csv('../datasets/glove.6B.100d.txt', sep=' ',  quoting=csv.QUOTE_NONE,names=[str(i) for i in range(101)])
    Eng_Spa,MAX_ENG,MAX_SPA=spli.get_train()
    Eng_array={'<sos>':0,'<pad>':1,'<eos>':3}
    MatrixEmbedded=pd.DataFrame([[0 for i in range(100)],[0.1 for i in range(100)],[0.2 for i in range(100)],[0.3 for i in range(100)]],columns=[str(i+1) for i in range(100)])
    len_Eng_array=4
    for i in range(Eng_Spa.shape[0]):
        for Eng_tokenized in Tokenization(Eng_Spa.iloc[i]['Eng']):
            if(Eng_array.get(Eng_tokenized,-1)==-1):
                df=(glove.loc[glove['0']==Eng_tokenized])
                if(df.empty):
                    Eng_array[Eng_tokenized]=2
                else:
                    Eng_array[Eng_tokenized]=len_Eng_array
                    df=df.drop(['0'],axis=1)
                    MatrixEmbedded=pd.concat([MatrixEmbedded,df])
                    len_Eng_array+=1
    MatrixEmbedded.to_csv('../datasets/MatrixEmbedded2.csv',index=False,header=False)

    with open('../datasets/Eng_dict2.pkl', 'wb') as f:
        pickle.dump(Eng_array, f)
            
    

def EmbedSpa():
    spli=Splitter()
    Eng_Spa,MAX_ENG,MAX_SPA=spli.get_train()
    Spa_array={'<sos>':0,'<pad>':1,'<eos>':3}
    len_Spa_array=4
    for i in range(Eng_Spa.shape[0]):
        for Spa_tokenized in Tokenization(Eng_Spa.iloc[i]['Spa']):
            if(Spa_array.get(Spa_tokenized,-1)==-1):
                Spa_array[Spa_tokenized]=len_Spa_array
                len_Spa_array+=1

    with open('../datasets/Spa_dict2.pkl', 'wb') as f:
        pickle.dump(Spa_array, f)


