import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from trainer import train_epoch,val_epoch
import os,sys
import yaml
from box import Box
from module import model_and_tokenizer

if len(sys.argv) == 1: config_path = './config.yaml'
else: config_path = sys.argv[1]
config = Box.from_yaml(filename=config_path)
saved_dir = config.main.saved_dir
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
with open(saved_dir+'/config.yaml','w+') as f:
    yaml.dump(config.to_dict(),f,default_flow_style=False)

MAX_LEN = config.data.kwargs.max_length
BATCH_SIZE = config.model.batch_size
EPOCHS = config.model.epochs
base_path = config.data.basepath

train = pd.read_csv(base_path+'train.tsv',delimiter='\t',header=None,names=['ids','label','alpha','sentence'])
val = pd.read_csv(base_path+'val.tsv',delimiter='\t',header=None,names=['ids','label','alpha','sentence'])

print('# of trainning sentence: {}'.format(train.shape[0]))
train_sents,train_labels = train.sentence.values,train.label.values
val_sents,val_labels = val.sentence.values,val.label.values

def sent_tokenize(sentences,tokenizer):
    ids,attention_masks = [],[]
    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(str(sent),add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt')
        ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])
    ids = torch.cat(ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)
    return ids,attention_masks

model,tokenizer = model_and_tokenizer(config.model.model_type,config.model.pretrain_weight)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_inputs,train_masks = sent_tokenize(train_sents,tokenizer)
val_inputs,val_masks = sent_tokenize(val_sents,tokenizer)
train_labels =torch.tensor(train_labels).to(torch.int64)
val_labels =torch.tensor(val_labels).to(torch.int64)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)


optimizer = AdamW(model.parameters(),lr = config.optimizer.kwargs.lr, eps = 1e-8)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1,1]).cuda())
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

# min_loss = 1e8
max_f1 = -1

for epoch in range(EPOCHS):
    train_loss,train_acc,train_f1 = train_epoch(model,train_dataloader,optimizer,criterion,scheduler)
    val_loss,val_acc,val_f1 = val_epoch(model,val_dataloader,optimizer,criterion,scheduler)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:3f}, Train F1: {train_f1:3f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:3f}, Val. F1: {val_f1:3f}')
    if val_f1 > max_f1:
        # torch.save(model.state_dict(), Save_path+'.pkl')
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(saved_dir)
        tokenizer.save_pretrained(saved_dir)
        # min_loss = val_loss
        max_f1 = val_f1
        print('model saved!')