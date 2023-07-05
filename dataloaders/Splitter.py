import pandas as pd
from dataloaders.Tokenization import *
from sklearn.model_selection import train_test_split

# def replacefunc(row):
#         row['Eng']=row['Eng'].replace(i,'')
#     print(row['Eng'])
#     return row

class Splitter:
    def __init__(self):
        self.Eng_Spa = pd.read_csv('./datasets/Eng_Spa.txt', sep='\t', names=['Eng','Spa','blaw']).iloc[:100000]
        self.Eng_Spa = self.Eng_Spa.drop('blaw', axis=1)
        unused=['!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_',"'",'{','|','}','~','\t','\n']
        for i in unused:
            self.Eng_Spa['Eng'] = self.Eng_Spa['Eng'].str.replace(i, '')
        self.Eng_Spa['Eng'] = self.Eng_Spa['Eng'].str.lower()
        self.Eng_Spa['Eng_len'] = self.Eng_Spa['Eng'].str.count('\s+')+1


        for i in unused:
            self.Eng_Spa['Spa'] = self.Eng_Spa['Spa'].str.replace(i, '')
        self.Eng_Spa['Spa'] = self.Eng_Spa['Spa'].str.lower()
        self.Eng_Spa['Spa_len'] = self.Eng_Spa['Spa'].str.count('\s+')+1
        self.Eng_Spa = self.Eng_Spa[self.Eng_Spa['Eng_len'].duplicated(keep=False)]
        

        y = self.Eng_Spa['Eng_len'].to_frame()
        X = self.Eng_Spa
        self.Eng_Spa_Train, self.Eng_Spa_test, __, _ = train_test_split(
                X, y,stratify=y, train_size=0.8,random_state=5)
        
        self.Eng_Spa_Train=self.Eng_Spa_Train.sort_values('Eng_len')
        
        self.MAX_ENG_LEN_train=self.Eng_Spa_Train.max()['Eng_len']
        self.MAX_SPA_LEN_train=self.Eng_Spa_Train.max()['Spa_len']

    def get_train(self):
        return self.Eng_Spa_Train, self.MAX_ENG_LEN_train, self.MAX_SPA_LEN_train
    def get_test(self):
        return self.Eng_Spa_test, self.MAX_ENG_LEN_train, self.MAX_SPA_LEN_train
