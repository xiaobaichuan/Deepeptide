import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.param_configs import Configs
from utils.data_process import training_data, predicted_data, create_data_loader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from utils.metrics import F1_score


###### 训练与测试
def train_epoch(model, data_loader, optimizer, scheduler, configs):
    # 训练模式
    model = model.train()
    loss_list = []
    for sample in tqdm(data_loader, 'train'):
        input_ids = sample['input_ids'].to(configs.device)
        attention_mask = sample['attention_mask'].to(configs.device)
        labels = sample['labels'].to(configs.device)
        # print(set([j.item() for i in list(sample['labels']) for j in i]))
        out = model(input_ids, attention_mask)
        loss = model.neg_log_likelihood(sentence_tensor=input_ids,
                                        label_tensor=labels,
                                        mask_tensor=attention_mask)
        loss_list.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(loss_list)


def eval_epoch(model, data_loader, configs):
    # 验证模式
    model = model.eval()
    with torch.no_grad():
        true_tag, predict_tag = [], []
        for sample in tqdm(data_loader, 'valid'):
            sentence_tensor = sample['input_ids'].to(configs.device)
            mask_tensor = sample['attention_mask'].to(configs.device)
            label_tensor = sample['labels'].to(configs.device)
            out = model(sentence_tensor, mask_tensor)
            for i in range(len(out)):
                tmp = []
                for j in range(len(out[i])):
                    tmp.append(configs.idx2tag[out[i][j]])
                predict_tag.append(tmp)
                true_tag.append(sample['tags'][i].split()[:len(out[i])])
        metrics = F1_score(true_tag, predict_tag)
    return list(map(float, [it[0] for it in metrics]))


###### 载入预设参数
torch.manual_seed(1)
configs = Configs()
device = configs.device
print('{} is available'.format(device))


#### 载入数据
root_path = '/home/project/'+configs.modelname
data_path, model_path = root_path+'/data/', root_path+'/models/model.pkl'  ##model_long/mix

if configs.is_training:
    if os.path.exists(data_path+'training_set/train.pt') and not configs.reload:  ##train_long/mix
        train_set = torch.load(data_path+'training_set/train.pt')
        valid_set = torch.load(data_path+'training_set/valid.pt')
    else:
        train_set, valid_set = training_data(data_path, configs)
    train_data_loader = create_data_loader(train_set, configs)
    valid_data_loader = create_data_loader(valid_set, configs)
elif not configs.is_predicting:
    if os.path.exists(data_path+'training_set/valid.pt'):  ##valid_long/mix
        valid_set = torch.load(data_path+'training_set/valid.pt')
    else:
        train_set, valid_set = training_data(data_path, configs)
    valid_data_loader = create_data_loader(valid_set, configs)
else:
    if os.path.exists(data_path+'test_set/prediction.pt'):
        pred_set = torch.load(data_path+'test_set/prediction.pt')
    else:
        pred_set = predicted_data(data_path, configs)
    pred_data_loader = create_data_loader(pred_set, configs)


######################################## main program ####################################
if configs.is_training:
    from models.TransformerNER import TransformerNER
    model = TransformerNER(configs).to(configs.device)
    
    ##### 根据实际需求选择需要微调的模型参数
    if configs.full_fine_tuning: ## 此种情况下，微调所有模型的参数
        # # # model.named_parameters(): [transformer, bilstm, hidden2tag, crf]
        trans_optimizer = list(model.trans.named_parameters())
        lstm_optimizer = list(model.rnn.named_parameters())
        classifier_optimizer = list(model.hidden2tag.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in trans_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': configs.weight_decay},
            {'params': [p for n, p in trans_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': configs.learning_rate * 5, 'weight_decay': configs.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': configs.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': configs.learning_rate * 10, 'weight_decay': configs.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': configs.learning_rate * 10, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': configs.learning_rate * 10}
        ]
    else: ## 自定义需要调整的范围
        param_optimizer = list(model.hidden2tag.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]},
            {'params': model.crf.parameters(), 'lr': configs.learning_rate * 5}]
    
    # 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.learning_rate, correct_bias=False)
    # 学习率指数衰减
    configs.train_size = len(train_set[0])
    train_steps_per_epoch = configs.train_size // configs.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(configs.num_epochs // 200) * train_steps_per_epoch,
                                                num_training_steps=configs.num_epochs * train_steps_per_epoch)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 开始训练过程
    EPOCHS = configs.num_epochs
    metric1_best, metric2_best, pat_cnt, cutoff = 0, 0, 0, configs.patience
    for epoch in range(EPOCHS):
        print('——'*10, f'Epoch {epoch + 1}/{EPOCHS}', '——'*10)
        train_loss = train_epoch(model, train_data_loader, optimizer, scheduler, configs)
        scheduler.step()
        print(f'Train loss : {round(train_loss, 2)}')

        recal, prec, F1 = eval_epoch(model, valid_data_loader, configs)
        print(f'valid recall : {round(recal, 2)}')
        print(f'valid precision : {round(prec, 2)}')
        print(f'valid F1 score : {round(F1, 2)}')
        metric1, metric2 = F1, prec
        if metric1-metric1_best>cutoff or (metric1-metric1_best>-cutoff and metric2-metric2_best>cutoff):
            metric1_best = metric1
            metric2_best = metric2
            print('save the best model until now.')
            torch.save(model, model_path)
            pat_cnt = 0
        else:
            pat_cnt += 1
        if epoch > configs.min_epoch_num and pat_cnt > configs.patience_num:
            break

elif not configs.is_predicting: # 验证模式
    model = torch.load(model_path)
    model = model.eval()
    res_file = open(root_path+'/result/vaild01.txt', 'w')  #### long/mix
    
    with torch.no_grad():
        predict_tag, true_tag = [], []
        for sample in tqdm(valid_data_loader, 'valid'):
            sentence_tensor = sample['input_ids'].to(configs.device)
            mask_tensor = sample['attention_mask'].to(configs.device)
            out = model(sentence_tensor, mask_tensor)
            sentence = sample['sentence']
            label = sample['tags']
            
            for i in range(len(out)):
                tmp = []
                for j in range(len(out[i])):
                    tmp.append(configs.idx2tag[out[i][j]])
                predict_tag.append(tmp)
                true_tag.append(label[i].split()[:len(out[i])])
                res_file.write('RawSequence: ' + sentence[i] + '\n')
                res_file.write('GroundTruth: ' + label[i] + '\n')
                res_file.write('PredictTags: ' + ' '.join(tmp) + '\n')
                
        recall, precision, F1 = F1_score(true_tag, predict_tag)
        res_file.write('recall(from +/-0 to +/-3): ' + ', '.join(recall) + '\n')
        res_file.write('precision(from +/-0 to +/-3): ' + ', '.join(precision) + '\n')
        res_file.write('F1(from +/-0 to +/-3): ' + ', '.join(F1) + '\n')
        
    res_file.close()
        
else: # 预测模式
    model = torch.load(model_path)
    print('Prediction process.')
    model = model.eval()
    res_file = open(root_path+'/result/prediction.txt', 'w') #### long/mix
    
    with torch.no_grad(): 
        for sample in tqdm(pred_data_loader, 'prediction'):
            sentence_tensor = sample['input_ids'].to(configs.device)
            mask_tensor = sample['attention_mask'].to(configs.device)
            out = model(sentence_tensor, mask_tensor)
            
            for i in range(len(out)):
                tmp = []
                for j in range(len(out[i])):
                    tmp.append(configs.idx2tag[out[i][j]])
                res_file.write('RawSequence: ' + sample['sentence'][i] + '\n')
                res_file.write('PredictTags: ' + ' '.join(tmp) + '\n')
    
    res_file.close()


