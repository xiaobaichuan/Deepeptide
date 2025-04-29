import torch
class Configs(object):
    def __init__(self):
        self.version = 'esm-lstm-crf'
        self.train_path = './data/embedding/train/'
        self.predict_path = './data/embedding/predict/'
        self.esm_path = './model/esm2_t33_650M_UR50D.pt'
        self.use_gpu = True
        
        self.reload = False
        self.is_training = False
        self.is_predicting = True
        
        self.num_models = 5
        self.repr_layers = 33
        self.pretrained_size = 1280
        
        self.cnn_dim = 128
        self.filter_size1 = 3
        self.filter_size2 = 5

        self.max_length = 1024
        self.hidden_size = 256 
        self.batch_size = 64
        self.num_epochs = 200
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.dropout_rate = 0.5
        self.hidden_size = 256
        self.num_layers = 2
        self.bidirectional = True
        self.split_ratio = 0.2
        self.patience = 0.0005
        self.patience_num = 5
        self.min_epoch_num = 10

        self.tag2idx = {"O": 0, "I": 1,
                        "B": 2, "E": 3}
        self.idx2tag = {0: "O", 1: "I",
                        2: "B", 3: "E"}
        
        self.bool_mask = {"T": True, "F": False, "1":True,"0":False}

        self.train_size = None
        self.valid_size = None
        self.num_class = len(self.tag2idx)
        self.word2idx = None
        self.vocab_len = None
        self.embedding_pretrained = None
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        
