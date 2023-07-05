
import torch
import torch.nn as nn
import random

class Decoder(nn.Module):
    def __init__(self, hidden_size,output_size,output_size_perline, teacherforcing=0.3,numlayer=3,dropout=0.8):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.output_size_perline = output_size_perline
        self.hidden_size = hidden_size
        self.teacherforcing=teacherforcing
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.batch=nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size,numlayer,batch_first=True,dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        outputscross = torch.zeros((input.shape[0], self.output_size_perline,self.output_size)).to('cuda')
        for i in range(input.shape[1]):
            if(i==0 or random.random()>self.teacherforcing):
                _,myinp=input[:,i].topk(1,axis=1)
            else:
                pass
                _,myinp=outputscross[:,i-1].topk(1,axis=1)
            embedded = self.embedding(myinp)
        #     # Run LSTM on embedded input
            output, hidden = self.lstm(embedded,hidden)
            # #     # Apply output layer and return output and hidden state
            output = self.out(output)
            output=torch.squeeze(output,dim=1)
            outputscross[:,i] = output

        return outputscross, hidden
    