import os
# import esm
import numpy as np
import torch
# import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.param_configs import Configs
from utils.data_process import create_data_loader, train_or_valid, prepare_predict_data
from utils.metrics import metrics
from transformers.optimization import get_cosine_schedule_with_warmup


def train_epoch(model, data_loader, optimizer, scheduler, configs):
    model = model.train()
    loss_list = []
    for sample in tqdm(data_loader, 'train'):
        sequence_tensor = sample['sequence_tensor'].to(configs.device)
        mask_tensor = sample['mask'].to(configs.device)
        label_tensor = sample['label_tensor'].to(configs.device)
        loss = model.neg_log_likelihood(sequence_tensor=sequence_tensor,
                                        label_tensor=label_tensor,
                                        mask_tensor=mask_tensor)
        loss_list.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(loss_list)


def eval_epoch(model, data_loader, configs):
    model = model.eval()
    with torch.no_grad():
        true_tag, predict_tag = [], []
        for sample in tqdm(data_loader, 'valid'):
            sequence_tensor = sample['sequence_tensor'].to(configs.device)
            mask_tensor = sample['mask'].to(configs.device)
            # label_tensor = sample['label_tensor'].to(configs.device)
            label = sample['label']
            out = model(sequence_tensor, mask_tensor)
            
            for i in range(len(out)):
                tmp = []
                for j in range(len(out[i])):
                    tmp.append(configs.idx2tag[out[i][j]])
                predict_tag.append(tmp)
                true_tag.append(list(label[i]))
        
        results = metrics(true_tag, predict_tag)
        for tolerance, scores in results.items():
            print(f"Tolerance: {tolerance}, Recall: {scores['recall']:.2f}, Precision: {scores['precision']:.2f}, F1 Score: {scores['f1_score']:.2f}")
        
    return [item[1] for item in results[0].items()]


torch.manual_seed(1)
configs = Configs()
device, version = configs.device, configs.version
print('{} is available'.format(device))


if configs.is_training:
    if os.path.exists('./data/train.pt') and not configs.reload:
        train_set = torch.load('./data/train.pt')
        valid_set = torch.load('valid.pt')
    else:
        train_set = train_or_valid('./data/train', configs)
        valid_set = train_or_valid('./data/valid', configs)
    train_data_loader = create_data_loader(train_set, configs)
    valid_data_loader = create_data_loader(valid_set, configs)
elif not configs.is_predicting:
    if os.path.exists('./data/valid.pt'):
        valid_set = torch.load('./data/valid.pt')
    else:
        valid_set = train_or_valid('./data/valid', configs)
    valid_data_loader = create_data_loader(valid_set, configs)


if configs.is_training:
    from models.dl_model import NER
    model = NER(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    train_steps_per_epoch = len(train_set[0]) // configs.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(configs.num_epochs//50)*train_steps_per_epoch,
                                                num_training_steps=configs.num_epochs*train_steps_per_epoch)

    EPOCHS, cutoff = configs.num_epochs, configs.patience
    metric1_best, metric2_best, pat_cnt = 0, 0, 0
    for epoch in range(EPOCHS):
        print('——'*10, f'Epoch {epoch + 1}/{EPOCHS}', '——'*10)
        train_loss = train_epoch(model, train_data_loader, optimizer, scheduler, configs)
        scheduler.step()
        print(f'Train loss : {round(train_loss, 2)}')
        
        recal, prec, F1 = eval_epoch(model, valid_data_loader, configs)

        metric1, metric2 = F1, prec
        if metric1-metric1_best>cutoff or (metric1-metric1_best>-cutoff and metric2-metric2_best>cutoff):
            metric1_best = metric1
            metric2_best = metric2
            print('save the best model until now.')
            torch.save(model, './models/DL_model.pkl')
            pat_cnt = 0
        else:
            pat_cnt += 1
        if epoch > configs.min_epoch_num and pat_cnt > configs.patience_num:
            break

else:
    model = torch.load('./models/DL_model.pkl').to(device)
    model = model.eval()
    print('Prediction process starts.')
    
    with open('./data/prediction.txt', 'w') as f:
        pred_set = prepare_predict_data('./data/prediction.fasta', configs)
        pred_data_loader = create_data_loader(pred_set, configs)
        with torch.no_grad():
            for sample in tqdm(pred_data_loader, 'prediction'):
                sequence_tensor = sample['sequence_tensor'].to(device)
                mask_tensor = sample['mask'].to(device)
                out = model(sequence_tensor, mask_tensor)
                
                for i in range(len(out)):
                    ID, sequence, tags = sample['ID'][i], sample['sequence'][i], ''
                    for j in range(len(out[i])):
                        tags += configs.idx2tag[out[i][j]]
                    f.write(f'{ID}\n{sequence}\n{tags}\n')
