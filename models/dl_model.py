import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
import esm


class NER(nn.Module):
    def __init__(self, configs):
        super(NER, self).__init__()
        self.configs = configs

        self.trans, _ = esm.pretrained.load_model_and_alphabet(configs.transformer_path)
        self.dropout = nn.Dropout()
        if configs.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.rnn = nn.LSTM(input_size=configs.pretrained_size,
                           hidden_size=configs.hidden_size,
                           num_layers=configs.num_layers,
                           batch_first=True,
                           bidirectional=configs.bidirectional)

        self.hidden2tag = nn.Linear(configs.hidden_size*self.num_directions, configs.num_class)
        self.crf = CRF(num_tags=configs.num_class, batch_first=True)

    def _init_hidden(self, batchs):  
        h_0 = torch.randn(self.configs.num_layers*self.num_directions, batchs, self.configs.hidden_size)
        c_0 = torch.randn(self.configs.num_layers*self.num_directions, batchs, self.configs.hidden_size)
        return self._make_tensor(h_0), self._make_tensor(c_0)

    def remove_cls(self, embedding, attention_mask, padding_value=0):
        embedding = embedding.cpu().tolist()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
        features = [torch.tensor(feat) for feat in features]
        return pad_sequence(features, batch_first=True, padding_value=padding_value)

    def _get_dnn_features(self, input_ids, attention_mask):
        results = self.trans(input_ids, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        trans_output = self._make_tensor(self.remove_cls(token_representations, attention_mask))
        trans_output = self.dropout(trans_output)
        
        h_0, c_0 = self._init_hidden(batchs=trans_output.size(0))
        rnn_output, (hidden, c) = self.rnn(trans_output, (h_0, c_0))
        output = self.hidden2tag(rnn_output)
        return output

    def neg_log_likelihood(self, sentence_tensor=None, label_tensor=None, mask_tensor=None):
        dnn_features = self._get_dnn_features(sentence_tensor, mask_tensor)
        label_tensor1 = self._make_tensor(self.remove_cls(label_tensor, mask_tensor))
        mask_tensor1 = self._make_tensor(self.remove_cls(mask_tensor, mask_tensor))
        return -self.crf(emissions=dnn_features, tags=label_tensor1, mask=mask_tensor1.bool())

    def _make_tensor(self, tensor):
        tensor_ret = tensor.to(self.configs.device)
        return tensor_ret

    def forward(self, input_data, mask_tensor):
        dnn_features = self._get_dnn_features(input_data, mask_tensor)
        mask_tensor1 = self._make_tensor(self.remove_cls(mask_tensor, mask_tensor))
        out = self.crf.decode(emissions=dnn_features, mask=mask_tensor1.bool())
        return out