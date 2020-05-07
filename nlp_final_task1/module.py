from torch import nn
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
                          XLNetConfig,XLNetForSequenceClassification,XLNetTokenizer,
                          RobertaConfig,RobertaForSequenceClassification,RobertaTokenizer
                          )
def model_and_tokenizer(model_type,pretrain_weight):
    MODEL_CLASSES = {
        'bert': (BertForSequenceClassification, BertTokenizer),
        'xlm': (XLMForSequenceClassification, XLMTokenizer),
        'xlnet': (XLNetForSequenceClassification,XLNetTokenizer),
        'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer),
        'robetra': (RobertaForSequenceClassification,RobertaTokenizer)
    }
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(
        pretrain_weight,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
    )
    tokenizer = tokenizer_class.from_pretrained(pretrain_weight,do_lower_case=True)
    return model,tokenizer

class custom_bert(nn.Module):
    def __init__(self,bertmodel,dropout,hidden_size):
        self.bert = bertmodel
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size,2)
        self.softmax = nn.LogSoftmax()
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        _,output = self.bert(input_ids,token_type_ids=None,attention_mask = attention_mask)
        output = self.dropout(output)
        logits = self.softmax(self.classifier(output))
        return logits