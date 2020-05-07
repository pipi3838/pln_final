import numpy as np
import torch
from module import model_and_tokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd
from tqdm import tqdm
from trainer import test_prediction
from box import Box
import yaml
import sys

config = Box.from_yaml(filename=sys.argv[1])

MAX_LEN = config.data.kwargs.max_length
BATCH_SIZE = config.model.batch_size
data_path = config.data.basepath
pretrain_weight = config.main.saved_dir
output_path = './ans_file/' + config.model.name +'_ans.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sent_tokenize(sentences,tokenizer):
    ids,attention_masks = [],[]
    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(str(sent),add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt')
        ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])
    ids = torch.cat(ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)
    return ids,attention_masks

test_data = pd.read_csv(data_path+'test.tsv', delimiter='\t', header=None,dtype={'id': str,'text':str}, names=['id', 'sentence'])
test_sents = test_data.sentence
test_ids = test_data.id.values
print('# test sentences: {}'.format(test_data.shape[0]))

model,tokenizer = model_and_tokenizer(config.model.model_type,pretrain_weight)
model.to(device)

test_inputs,test_masks = sent_tokenize(test_sents,tokenizer)
test_input = TensorDataset(test_inputs,test_masks)
test_sampler = SequentialSampler(test_input)
test_dataloader = DataLoader(test_input, sampler=test_sampler, batch_size=BATCH_SIZE)

prediction = test_prediction(model,test_dataloader)
np.save('./'+config.model.name+'_pred.npy',prediction)

out = open(output_path,'w')
out.write('Index,Gold\n')

pred = np.argmax(prediction,axis=1)

for index,p in zip(test_ids,pred):
    out.write("{},{}\n".format(index,p))