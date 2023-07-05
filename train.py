import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import pandas as pd
from tqdm import tqdm
from dataloaders.DataLoader import *
from nets.model import *
import os





    

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
    return model, optimizer


def accuracy(output, target):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
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
        return res

from torchmetrics.functional import bleu_score

def valuetokey(dataloader,values):
    keys=[]
    res = dict((v,k) for k,v in dataloader.Spa_array.iteritems())
    for i in values:
        keys.append(res[i].item())
    return ' '.join(keys)

def bleu(dataloader,output, target):
    output=valuetokey(dataloader,output)
    target=valuetokey(dataloader,target)
    return bleu_score(output,target)

def train(
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
    batch_size,ignore_in,epochadd=0
):
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_in)

    # optimzier
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
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"
        
        
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
            loss=0
            optimizer.zero_grad()
            for i in range(labels_pred.shape[1]):
                loss1 = criterion(labels_pred[:,i],labels2[:,i])
                if(not torch.isnan(loss1)):
                    loss += loss1
            loss.backward()
            optimizer.step()
            acc1 = accuracy(labels_pred, labels2)
            top1_acc_train.update(acc1[0], images.size(0))
            loss_avg_train.update(loss.item(), images.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch+epochadd,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_top1_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch+epochadd}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch+epochadd}.ckpt",
                model=model,
                optimizer=optimizer,
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
                loss=0
                optimizer.zero_grad()
                for i in range(labels_pred.shape[1]):
                    loss1 = criterion(labels_pred[:,i],labels2[:,i])
                    if(not torch.isnan(loss1)):
                        loss += loss1
                acc1 = accuracy(labels_pred, labels2)
                top1_acc_val.update(acc1[0], images.size(0))
                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch+epochadd,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_top1_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_top1_acc_till_current_batch":top1_acc_val.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch+epochadd}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )
        lr_scheduler.step()
        report.to_csv(f"{report_path}/{model_name}_report.csv")
    return model, optimizer, report


from utlis import Read_yaml
yml=Read_yaml.Getyaml()



batch_size = yml['batch_size']
epochs = yml['num_epochs']
learning_rate = yml['learning_rate']
gamma=yml['gamma']
step_size=yml['step_size']
ckpt_save_freq = yml['ckpt_save_freq']

ckpt_save_path = yml['ckpt_save_path']
report_path = yml['report_path']
ckpt_path = yml['ckpt_path']



num_layer = yml['num_layer']
teacher_forcing = yml['teacher_forcing']
dropout = yml['dropout']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_dataset = MyDataset(train=True)
test_dataset = MyDataset(train=False)


ignore_in=1
custom_model = Seq2SeqTrans(num_layer,teacher_forcing,dropout)



train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)





trainer = train(
    train_loader=train_loader,
    val_loader=test_loader,
    model = custom_model,
    model_name="Custom model",
    epochs=1,
    learning_rate=learning_rate,
    gamma = gamma,
    step_size = step_size,
    device=device,
    load_saved_model=False,
    ckpt_save_freq=ckpt_save_freq,
    ckpt_save_path=ckpt_save_path,
    ckpt_path=ckpt_path,
    report_path=report_path,
    batch_size=batch_size,ignore_in=ignore_in
)







train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)





trainer = train(
    train_loader=train_loader,
    val_loader=test_loader,
    model = custom_model,
    model_name="Custom model",
    epochs=epochs,
    learning_rate=learning_rate,
    gamma = gamma,
    step_size = step_size,
    device=device,
    load_saved_model=False,
    ckpt_save_freq=ckpt_save_freq,
    ckpt_save_path=ckpt_save_path,
    ckpt_path=ckpt_path,
    report_path=report_path,
    batch_size=batch_size,ignore_in=ignore_in,epochadd=1
)
