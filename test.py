import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import pandas as pd
from tqdm import tqdm
from dataloaders.DataLoader import *
from nets.model import *
import os




def change_to_word(output,target):
    _, pred = output.topk(1, 2, True, True)
    pred=torch.unsqueeze(pred,dim=2)
    pred2 = target
    correct=[]
    for i in range(pred.shape[1]):
        correct.append((pred[0][i][0][0],pred2[0][i]))
    print(correct)


def accuracy(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 2, True, True)
        pred=torch.squeeze(pred,dim=2)
        
        pred2 = target
        total=0
        correct=0
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if(pred2[i,j]!=1):
                    total+=1
                    if(pred2[i,j]==pred[i,j]):
                        correct+=1
        res=[]
        res.append(correct/total*100)
        return res[0]
    

class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    print("loaded")
    return model, optimizer


from torchmetrics.functional import bleu_score

def valuetokey(dataloader,values):
    keys=[]
    res = dict((v,k) for k,v in dataloader.Spa_array.items())
    res[2]='<unk>'
    for i in values:
        if(keys==[]):
            keys.append(res[i.item()])
        else:
            if(keys[-1]!='<eos>'):
                keys.append(res[i.item()])
    return ' '.join(keys)

def bleu(dataloader,output, target):
    bl=[]
    _,output=output.topk(1,2)
    for i in range(output.shape[0]):
        aa=valuetokey(dataloader,target[i])
        bb=valuetokey(dataloader,output[i])
        bl.append(bleu_score([aa],[bb]))
    return np.mean(np.array(bl))

def test(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
    batch_size,train_dataset,test_dataset
):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch"])


    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        bl_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        bl_avg_val = AverageMeter()
        model.eval()
        mode = "train"
        with torch.no_grad():
            loop_train = tqdm(
                enumerate(train_loader, 1),
                total=len(train_loader),
                desc="train",
                position=0,
                leave=True)
            for batch_idx, (images, labels,labels2) in loop_train:
                images = images.to(device)
                labels = labels.to(device)
                labels2 = labels2.type(torch.LongTensor).to(device)
                labels_pred = model(images,labels)
                # change_to_word(labels_pred, labels2)
                acc=accuracy(labels_pred, labels2)
                bl=bleu(train_dataset,labels_pred, labels2)
                top1_acc_train.update(acc, images.size(0))
                bl_avg_train.update(bl, images.size(0))                
                loop_train.set_description(f"train - iteration : {epoch}")
                loop_train.set_postfix(
                    bl_batch="{:.4f}".format(bl),
                    avg_train_bl_till_current_batch="{:.4f}".format(bl_avg_train.avg),
                    top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                    refresh=True,
                )
                


        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (images, labels,labels2) in loop_val:
                images = images.to(device)
                labels = labels.to(device)
                labels2 = labels2.type(torch.LongTensor).to(device)
                labels_pred = model(images,labels)
                acc=accuracy(labels_pred, labels2)
                bl=bleu(test_dataset,labels_pred, labels2)
                top1_acc_val.update(acc, images.size(0))
                bl_avg_val.update(bl, images.size(0))
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    bl_batch="{:.4f}".format(bl),
                    avg_val_bl_till_current_batch="{:.4f}".format(bl_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )


    return model



from utlis import Read_yaml
yml=Read_yaml.Getyaml()



batch_size = yml['batch_size']
ckpt_save_path = yml['ckpt_save_path']
report_path = yml['report_path']
ckpt_path = yml['ckpt_path']



num_layer = yml['num_layer']
teacher_forcing = yml['teacher_forcing']
dropout = yml['dropout']



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_dataset = MyDataset(train=True)
test_dataset = MyDataset(train=False)



custom_model = Seq2SeqTrans(num_layer,teacher_forcing,dropout)




train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)





trainer = test(
    train_loader=train_loader,
    val_loader=test_loader,
    model = custom_model,
    model_name="Custom model",
    epochs=1,
    learning_rate=1,
    gamma = 1,
    step_size = 1,
    device=device,
    load_saved_model=True,
    ckpt_save_freq=1,
    ckpt_save_path="./ckpts/",
    ckpt_path="./ckpts/check.ckpt",
    report_path="./reports/",
    batch_size=batch_size,
    train_dataset=train_dataset,test_dataset=test_dataset)