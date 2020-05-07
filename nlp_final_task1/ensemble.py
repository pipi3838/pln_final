import numpy as np

pred1 = np.load('bertBase_pred.npy')
pred2 = np.load('bertBase_uncased_pred.npy')
pred3 = np.load('xlnet_maxLen_128_weight0.1_pred.npy')
pred4 = np.load('bertLarge_uncased_pred.npy')
pred5 = np.load('xlnet-cased_pred.npy')

test_ids = np.load('test_ids.npy',allow_pickle=True)

pred = pred1 + pred2 + pred3 + pred4 + pred5
arg_pred = np.argmax(pred,axis=1)

out = open('ensemble5_ans.csv','w')
out.write('Index,Gold\n')

for index,p in zip(test_ids,arg_pred):
    out.write("{},{}\n".format(index,p))