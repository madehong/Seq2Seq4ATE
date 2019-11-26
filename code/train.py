import argparse
import torch
import time
import json
import numpy as np
import math
import random
from config import args
import sys
args.domain = sys.argv[1]
args.model_name = args.domain

if args.domain == 'laptop':
    seed = 14
elif args.domain == 'restaurant':
    seed = 34
args.seed = seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from model import Encoder, Decoder, Seq2Seq

def batch_generator(X, y, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len=np.sum(X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_X_len.argsort()[::-1]
        batch_X_len=batch_X_len[batch_idx]
        
        batch_X_mask=(X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)

        batch_X=X[offset:offset+batch_size][batch_idx] 
        batch_y=y[offset:offset+batch_size][batch_idx]

        batch_X = torch.from_numpy(batch_X).long().cuda()
        batch_X_mask=torch.from_numpy(batch_X_mask).long().cuda()
        batch_y = torch.from_numpy(batch_y).long().cuda()
        
        if len(batch_y.size() )==2 and not crf:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len)

def valid_loss(model, valid_X, valid_y, batch_size=16, crf=False):
    model.eval()
    losses=[]
    for batch in batch_generator(valid_X, valid_y, batch_size=batch_size, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len = batch
        loss=model(batch_valid_X, batch_valid_X_len, batch_valid_y)
        #losses.append(loss.data[0])
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

def train(train_X, train_y, valid_X, valid_y, model, model_fn, optimizer, parameters, epochs=200, batch_size=128, crf=False):
    best_loss=float("inf") 
    valid_history=[]
    train_history=[]
    for epoch in range(epochs):
        i = 0
        for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
            batch_train_X, batch_train_y, batch_train_X_len = batch
            loss=model(batch_train_X, batch_train_X_len, batch_train_y)
            print('Epoch: {}; batch: {}; loss:{}'.format(epoch, i, loss))
            i += 1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        loss=valid_loss(model, train_X, train_y, batch_size, crf=crf)
        train_history.append(loss)
        loss=valid_loss(model, valid_X, valid_y, batch_size, crf=crf)
        valid_history.append(loss)        
        if loss<best_loss:
            best_loss=loss
            torch.save(model, model_fn) 
        shuffle_idx=np.random.permutation(len(train_X) )
        train_X=train_X[shuffle_idx]
        train_y=train_y[shuffle_idx]
    model=torch.load(model_fn) 
    return train_history, valid_history

def run(domain, data_dir, model_dir, valid_split, epochs, lr, dropout, batch_size=128, model_name=''):
    gen_emb=np.load(data_dir+"gen.vec.npy")
    ae_data=np.load(data_dir+domain+".npz")

    idx = np.arange(ae_data['train_X'].shape[0])
    np.random.shuffle(idx)

    train_X = ae_data['train_X']
    train_y = ae_data['train_y']
    
    valid_X=train_X[-valid_split:]
    valid_y=train_y[-valid_split:]

    train_X=train_X[:-valid_split]
    train_y=train_y[:-valid_split]

    encoder = Encoder(gen_emb, args)
    decoder = Decoder(args)
    model   = Seq2Seq(encoder, decoder, args)
    model.cuda()
    print(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters, lr=lr)
    train_history, valid_history=train(train_X, train_y, valid_X, valid_y, model, model_dir+model_name, optimizer, parameters, epochs, batch_size, crf=False)
    

run(args.domain, args.data_dir, args.model_dir, args.valid, args.epochs, args.lr, args.dropout, args.batch_size, args.model_name)

