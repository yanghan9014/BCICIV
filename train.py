from x_transformers import TransformerWrapper, Decoder
from x_transformers.continuous_autoregressive_wrapper import ContinuousAutoregressiveWrapper

import argparse
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from metrics import Metrics

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1000
SEQ_LEN = 1000

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default="data/npz/sub3_comp.npz",
                        help="One npz file")
    args = parser.parse_args()
    return args

class BID_Dataset(Dataset):
  def __init__(self, path, seq_len, train = True, full = True):
    if train:
        self.X = torch.tensor(dict(np.load(path, allow_pickle=True))['train_data']).cuda()
        self.y = torch.tensor(dict(np.load(path, allow_pickle=True))['train_dg']).cuda()
    else:
        self.X = torch.tensor(dict(np.load(path, allow_pickle=True))['test_data']).cuda()
        self.y = torch.tensor(dict(np.load(path, allow_pickle=True))['test_dg']).cuda()

    self.seq_len = seq_len
    self.full = full
    
  def __getitem__(self, index):
    if self.full:
        parital_X = self.X[index * self.seq_len : (index + 1) * self.seq_len]
        return self.X.cuda()

    rand_start = torch.randint(0, self.X.shape[0] - self.seq_len + 1, (1,))
    parital_X = self.X[rand_start: rand_start + self.seq_len].long()
    parital_y = self.y[rand_start: rand_start + self.seq_len].long()
    return parital_X.cuda(), parital_y.cuda()
    
  def __len__(self):
    return self.X.shape[0] // self.seq_len



# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def train():
    args = parse()
    metrics = Metrics()
    
    train_dataset   = BID_Dataset(args.path, SEQ_LEN, train=True, full=False)
    val_dataset     = BID_Dataset(args.path, SEQ_LEN, train=False, full=False)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

    train_dataset_full = BID_Dataset(args.path, SEQ_LEN, train=True, full=True)
    val_dataset_full   = BID_Dataset(args.path, SEQ_LEN, train=False, full=True)
    train_loader_full  = cycle(DataLoader(train_dataset_full, batch_size = BATCH_SIZE, drop_last = False))
    val_loader_full    = cycle(DataLoader(val_dataset_full, batch_size = BATCH_SIZE, drop_last = False))


    # instantiate GPT-like decoder model
    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = SEQ_LEN,
        attn_layers = Decoder(dim = 5, depth = 6, heads = 8)
    )
    model = ContinuousAutoregressiveWrapper(model)
    model.cuda()

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        model.train()
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()

        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            pred = None
            for data in train_loader_full:
                if pred is None:
                    pred = model.generate(data, SEQ_LEN)
                else:
                    pred = torch.cat((pred, model.generate(data, SEQ_LEN)), dim=0)
            print("Correlation coefficient:", metrics(pred, train_dataset_full.y))
if __name__ == '__main__':
    train()