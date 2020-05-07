import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model,train_dataloader,optimizer,criterion,scheduler):
    train_loss,train_acc = 0,0
    store_label,store_pred = np.array([]),np.array([])
    model.train()
    sigmoid = nn.Sigmoid()
    for batch in tqdm(train_dataloader):
        b_input_ids,b_mask,b_label = tuple(t.to(device) for t in batch)
        model.zero_grad()
        loss,logits = model(b_input_ids,token_type_ids=None,attention_mask=b_mask,labels=b_label)
        logits = sigmoid(logits)
        # label_2d = torch.stack([1-b_label,b_label],axis=1).to(torch.float)
        # loss = criterion(logits,label_2d)
        train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        b_label = b_label.to('cpu').numpy()
        logits = np.argmax(logits,axis=1)
        # print(b_label,logits)
        train_acc += np.sum(logits == b_label) / len(b_label)
        store_label = np.concatenate((store_label,b_label.reshape(-1)))
        store_pred = np.concatenate((store_pred,logits.reshape(-1)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    _,_,train_f1,_ = precision_recall_fscore_support(store_label,store_pred,labels=[0,1],average='weighted')
    return train_loss,train_acc,train_f1

def val_epoch(model,val_dataloader,optimizer,criterion,scheduler):
    model.eval()
    val_loss,val_acc = 0,0
    store_label,store_pred = np.array([]),np.array([])
    sigmoid = nn.Sigmoid()
    for batch in val_dataloader:
        b_input_ids,b_mask,b_label = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss,logits = model(b_input_ids,token_type_ids=None,attention_mask=b_mask,labels=b_label)
            logits = sigmoid(logits)
            # label_2d = torch.stack([1-b_label,b_label],axis=1).to(torch.float)
            # loss = criterion(logits,label_2d)
            val_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            b_label = b_label.to('cpu').numpy()
            logits = np.argmax(logits,axis=1)
            store_label = np.concatenate((store_label,b_label.reshape(-1)))
            store_pred = np.concatenate((store_pred,logits.reshape(-1)))
            val_acc += np.sum(logits == b_label) / len(b_label)
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    _,_,val_f1,_ = precision_recall_fscore_support(store_label,store_pred,labels=[0,1],average='weighted')
    return val_loss,val_acc,val_f1

def test_prediction(model,test_dataloader):
    model.eval()
    prediction = []

    for batch in tqdm(test_dataloader):
        b_id,b_mask = batch[0].to(device),batch[1].to(device)

        with torch.no_grad():
            output = model(b_id,token_type_ids=None,attention_mask=b_mask)
            logits = output[0]
            logits = logits.detach().cpu().numpy()
            prediction.append(logits)
    prediction = np.vstack(prediction)
    return prediction