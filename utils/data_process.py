import torch
import esm
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, data=None, configs=None):
        self.configs = configs
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        sample = {}
        if len(self.data) == 5:
            sample['sequence'] = self.data[0][item]
            sample['label'] = self.data[1][item]
            sample['sequence_tensor'] = self.data[2][item]
            sample['label_tensor'] = self.data[3][item]
            sample['mask'] = self.data[4][item]
        else:
            sample['ID'] = self.data[0][item]
            sample['sequence'] = self.data[1][item]
            sample['sequence_tensor'] = self.data[2][item]
            sample['mask'] = self.data[3][item]
        return sample


def create_data_loader(data, configs):
    ds = Dataset(data=data, configs=configs)
    return DataLoader(ds, batch_size=configs.batch_size)


def is_all_BIE(sequence):  ## sequence is the variable recording 'BIEO'
    BIE_length = 0
    start = None
    for i, label in enumerate(sequence):
        if label.startswith("B"):  # 实体开始
            start = i
        elif label.startswith("E") and start is not None:  # 实体结束
            BIE_length += i-start+1
            start = None
        elif label == "O":  # 遇到外部标记，重置开始位置
            start = None
    if BIE_length == len(sequence):
        return True
    else:
        return False


def Encoding(data, batch_converter, model):
    ## data = [(ID, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    res = {}
    for i in range(len(data)):
        seqlength = len(data[i][1])
        token_representations = results["representations"][33][i][1:seqlength+1].numpy()
        res[data[i][0]] = token_representations
    return res


def prepare_labeled_data(data_path, configs):
    with open(data_path, 'r') as f:
        tmp = f.read().split()
        id_list = [item[1:] for item in tmp[::4]]
        sequences, tags, mask_raw = tmp[1::4], tmp[2::4], tmp[3::4]
    
    _, alphabet = esm.pretrained.load_model_and_alphabet(configs.esm_path)
    batch_converter = alphabet.get_batch_converter()
    
    max_len = max([len(seq) for seq in sequences])
    labels, masks = [], []
    
    for i in range(len(id_list)):
        pad_length = max_len - len(tags[i])
        labels.append([configs.tag2idx[tag] for tag in tags[i]]+[0]*pad_length)
        masks.append([configs.bool_mask[m] for m in mask_raw[i]]+[False]*pad_length)
        # seq_masks.append([True]*len(tags[i])+[False]*pad_length)
    
    __, _, seq_id = batch_converter(list(zip(range(len(sequences)), sequences)))
    
    ret = [sequences, tags, seq_id, torch.tensor(labels), torch.tensor(masks)] 
    
    return ret


def prepare_predict_data(data_path, configs):
    with open(data_path, 'r') as f:
        tmp = f.read().split()
        id_list = [item[1:] for item in tmp[::2]]
        sequences = tmp[1::2]
    
    _, alphabet = esm.pretrained.load_model_and_alphabet(configs.esm_path)
    batch_converter = alphabet.get_batch_converter()
    
    max_len = max([len(seq) for seq in sequences])
    masks = []
    
    for i in range(len(id_list)):
        pad_length = max_len - len(sequences[i])
        masks.append([True]*len(sequences[i])+[False]*pad_length)
    
    __, _, seq_id = batch_converter(list(zip(range(len(sequences)), sequences)))
    
    ret = [id_list, sequences, torch.tensor(seq_id), torch.tensor(masks)]

    return ret


def training_data(data_path, configs):
    data_set = prepare_labeled_data(f'{data_path}.txt', configs)
    torch.save(data_set, f'{data_path}.pt')
    print(f'Total of {len(data_set[0])} items in the dataset.')
    return data_set


def predicted_data(data_path, configs):
    data_set = prepare_predict_data(f'{data_path}.txt', configs)
    torch.save(data_set, f'{data_path}.pt')
    print(f'Total of {len(data_set[0])} items in the dataset.')
    return data_set

