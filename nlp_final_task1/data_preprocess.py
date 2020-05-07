import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
import os

root_path = '/nfs/nas-5.1/wbcheng/nlp_final/'
data = pd.read_csv(root_path+'train.csv',sep=';',engine='python')
bert_data = pd.DataFrame({'id': np.arange(len(data)),
                            'label': data['Gold'],
                            'alpha': ['a'] * data.shape[0],
                            'text': data['Text']
})
bert_train,bert_val = train_test_split(bert_data,test_size=0.2,stratify=bert_data.label.values)
bert_train.to_csv(root_path+'train.tsv',sep='\t',index=False,header=False)
bert_val.to_csv(root_path+'val.tsv',sep='\t',index=False,header=False)

test_data = pd.read_csv(root_path+'test.csv',sep=';',dtype={'Index': str,'Text':str})
bert_test = pd.DataFrame({'id': test_data['Index'],
                            'text': test_data['Text']
})
bert_test.to_csv(root_path+'test.tsv',sep='\t',index=False,header=False)