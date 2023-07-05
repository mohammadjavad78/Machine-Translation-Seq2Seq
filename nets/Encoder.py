import torch
import torch.nn as nn




class Encoder(nn.Module):
    def __init__(self,embedding_matrix,numlayer,dropout):
        super(Encoder, self).__init__()
        hidden_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False
        # self.batch=nn.BatchNorm2d(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,numlayer,batch_first=True,dropout=dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        # embedded = self.batch(embedded)
        output, hidden = self.lstm(embedded)
        return output, hidden
