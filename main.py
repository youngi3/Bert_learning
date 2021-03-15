# 统一导入工具包
import pandas as pd
import csv
import transformers
import torch
from transformers import BertPreTrainedModel, BertModel
from torch import nn
import numpy as np
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Kagging debug
kagging =True
#2.1数据说明
pd_table = pd.read_csv('./datasets/datasets/raw/WikiQA-train.tsv',encoding="utf-8",sep = '\t')
pd_table

#2.2数据加载
def load(filename):
    result = []
    with open(filename, 'r', encoding = 'utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t', quotechar = '"')
        next(spamreader, None)
        for row in spamreader:
            # print(row)
            res = {}
            res['question'] = str(row[1])
            res['answer'] = str(row[5])
            res['label'] = int(row[6])
            result.append(res)
    return result                        

train_file = load('./datasets/datasets/raw/WikiQA-train.tsv')
valid_file = load('./datasets/datasets/raw/WikiQA-dev.tsv')
test_file = load('./datasets/datasets/raw/WikiQA-test.tsv')

print(len(train_file))
print(len(valid_file))
print(len(test_file))
# Kagging debug
if kagging:
    train_file = train_file[:int(len(train_file)*0.01)]
    valid_file = valid_file[:int(len(valid_file)*0.01)]
    test_file = test_file[:int(len(test_file)*0.01)]
    print(len(train_file))
    print(len(valid_file))
    print(len(test_file))

train_file[:10]

#2.3数据标准化
#2.3.1补齐
def padding(sequence, max_length, pad_token = 0):
    padding_length = max_length - len(sequence)
    return sequence + [pad_token] * padding_length

#转化为Bert标准输入
model_path = './datasets/datasets/models/bert-pretrain/'
tokenizer = transformers.BertTokenizer.from_pretrained(model_path,do_lower_case=True)
# max_length = 512 
max_length = 64 # Kagging debug
device = torch.device('cpu')

def tokenize(data, max_length, tokenizer, device):
    res = []
    for triple in data:
        inputs = tokenizer.encode_plus(
            triple['question'],
            triple['answer'],
            add_special_tokens = True,
            max_length = max_length,
            trunction = True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs['token_type_ids']
        attention_mask = [1] * len(input_ids)
        input_ids = padding(
            input_ids,
            max_length,
            pad_token = 0
        )
        attention_mask = padding(
            attention_mask,
            max_length,
            pad_token = 0
        )
        token_type_ids = padding(
            token_type_ids,
            max_length,
            pad_token = 0
        )
        label = triple['label']
        res.append((input_ids, attention_mask, token_type_ids, label))
    all_input_ids = torch.tensor([x[0] for x in res], dtype = torch.int64, device = device)
    all_attention_mask = torch.tensor([x[1] for x in res], dtype = torch.int64, device = device)
    all_token_type_ids = torch.tensor([x[2] for x in res], dtype = torch.int64, device = device)
    all_labels = torch.tensor([x[3] for x in res], dtype = torch.int64, device = device)
    return torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
test = tokenize(test_file, max_length, tokenizer, device)
# next(iter(test))
print(test[0])

#3模型定义
#3.1定义模型
config = transformers.BertConfig.from_pretrained(model_path)

class BertQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQA, self).__init__(config)
        self.num_labels = config.num_labels   #2
        self.bert = BertModel(config)
        
        for p in self.parameters():
            p.requires_grad = False
        
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')

        self.init_weights()
    
    def forward(self, input_ids = None, attention_mask = None, 
                token_type_ids = None, position_ids = None,
                 head_mask = None, inputs_embeds = None,
                  labels = None):
        outputs = self.bert(input_ids,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            position_ids = position_ids,
                            head_mask = head_mask,
                            inputs_embeds = inputs_embeds)
        logits = self.qa_outputs(outputs[0][:, 0, :]).squeeze()
        predicted_labels = nn.functional.softmax(logits, dim = -1)

        if labels is not None:
            loss = self.loss_fn(predicted_labels, labels)
            return loss, predicted_labels
        else:
            return predicted_labels

model = BertQA.from_pretrained(model_path, config = config)

model.to(device)

#4模型训练
#4.1数据生成器
batch_size = 4
train_dataset = tokenize(train_file, max_length, tokenizer, device)
train_sample = torch.utils.data.RandomSampler(train_dataset)
train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler = train_sample, batch_size = batch_size)

#4.2训练过程
epoch_num = 1
learning_rate = 1e-5
adam_epsilon = 1e-8
save_path = './temp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

optimizer = transformers.AdamW(filter(lambda p:p.requires_grad, model.parameters()),lr = learning_rate, eps = adam_epsilon)

loss_list = []
for epoch in range(epoch_num):
    print("Training epoch {}".format(epoch+1))
    pbar = tqdm(train_dataloader)
    print("pbar length is: {}".format(len(pbar)))
    for step, batch in enumerate(pbar):
        if step == 10:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_param.bin'))
            print("Model Saved")
            print("Stopted Early")
        model.train()
        model.zero_grad()
        inputs = {
            'input_ids':batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }
        outputs = model(**inputs)
        loss, results = outputs
        loss.backward() 
        optimizer.step()
        loss_list.append(loss.item())
        pbar.set_description('batch loss {:.3f}',format(loss.item()))

# plt.plot(loss_list)
# plt.show()

#5模型测试
def AP(output, target):
    output = torch.tensor(output,dtype = torch.float)
    target = torch.tensor(target,dtype = torch.float)

    _, indexes = torch.sort(output, descending = True)
    target = target[indexes].round()
    total = 0.
    for i in range(len(output)):
        index = i+1
        if target[i]:
            total +=target[:index].sum().item()/index
        return total/target.sum().item()

def MAP(outputs, targets):
    assert(len(outputs) == len(targets))
    res = []
    for i in range(len(outputs)):
        res.append(AP(outputs[i], targets[i]))
    return np.mean(res)

def RR(output, target):
    _, indexes = torch.sort(output, descending = True)
    best = target[indexes].nonzero().squeeze().min().item()
    return 1.0/(best+1)

def MRR(outputs, targets):
    assert(len(outputs) == len(targets))
    res = []
    for i in range(len(outputs)):
        res.append(RR(outputs[i], targets[i]))
    return np.mean(res)

test_dataset = tokenize(test_file,max_length,tokenizer,device)
# 创建Sampler
test_sampler = torch.utils.data.RandomSampler(test_dataset)
# 通过Dataset和Sampler创建dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model.load_state_dict(torch.load('temp/best_param.bin'))

tot_right = 0
tot = 0
y_hats = []
y_gold = []
os.path.join(save_path, 'result.txt')
# result_file = open('./result/result.txt','w',encoding='utf-8')
result_file = open('./temp/reslut.txt','w',encoding='utf-8')
for step, batch in enumerate(tqdm(test_dataloader)):
    #检测label是否全是0，即无正确答案，这样的样例无法计算MAP和MRR
    all_zero = True
    for i in batch[3]:
        if i!=0:
            all_zero = False
            break
    if all_zero == True:
        continue # 跳过这个样例
    model.eval()
    with torch.no_grad():
        if step == 2: # 提前停止
            break
        inputs = {
            'input_ids':      batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2]
        }
        y_hat = model(**inputs)
        y_hat = torch.argmax(y_hat, dim=-1)
        y_wrt_list = [str(int(yh))+'\n' for yh in y_hat.data]
        result_file.writelines(y_wrt_list)
        tot_right += (y_hat == batch[3]).sum()
        tot += len(batch[3])
        y_gold.append(batch[3])
        y_hats.append(y_hat)
print("tot_right is: {}".format(tot_right))
print("tot is: {}".format(tot))
print("Accuracy is {}".format(float(tot_right) / tot))
print("MAP is {}".format(MAP(y_hats,y_gold)))
print("MRR is {}".format(MRR(y_hats,y_gold)))
result_file.close()