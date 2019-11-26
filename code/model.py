#encoding=utf-8
import argparse
import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F
from config import args


seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


#########################################
##### Seq2Seq for aspect extraction #####
#########################################

class Encoder(torch.nn.Module):
    def __init__(self, gen_emb, args, dropout=0.5):
        super(Encoder, self).__init__()
        self.args = args
        
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        
        self.dropout = torch.nn.Dropout(args.dropout)

        self.rnn = torch.nn.GRU(gen_emb.shape[1], args.encoder_hidden_size, args.encoder_num_layers, batch_first=False, bidirectional=args.bidirectional)

    def forward(self,x, x_len):
        #encode
        x_emb=self.gen_embedding(x)
        x_emb=self.dropout(x_emb)
        total_length = x_emb.size(0)
        packed_emb=torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_len)
        encoder_outputs, encoder_hidden = self.rnn(packed_emb)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs, total_length=total_length)

        return encoder_outputs, encoder_hidden

def index(i, max_len):
    if i == 0:
        tmp = [x_ + 2 for x_ in range(max_len)]
    else:
        tmp = [x_ + 1 + 2 for x_ in range(i)][::-1] + [x_ + 2 for x_ in range(max_len - i)]
    #log2
    tmp = np.log2(tmp)
    tmp = tmp.reshape((1, max_len))
    return torch.from_numpy(tmp).float().cuda()

class Attention(torch.nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_hidden_size):
        super(Attention, self).__init__()
        self.attn = torch.nn.Linear(encoder_hidden_size*2 + decoder_hidden_size, attention_hidden_size)
        self.v = torch.nn.Parameter(torch.rand(attention_hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, attention_vector, encoder_outputs, time_step, max_len, mask=None):
        
        timestep = encoder_outputs.size(0)
        h = attention_vector.repeat(timestep, 1, 1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        score = self.score(h, encoder_outputs)
        weight= index(time_step, max_len)
        score = score / weight
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        return F.softmax(score, dim=1).unsqueeze(1)
    
    def score(self, h, encoder_outputs):
        concat = torch.cat([h, encoder_outputs], 2)
        s = self.attn(concat)
        s = s.transpose(1,2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        s = torch.bmm(v, s)
        return s.squeeze(1)


class Decoder(torch.nn.Module):
    def __init__(self, args, num_classes=3, dropout=0.2):
        super(Decoder, self).__init__()
        self.args = args
        
        self.label_embedding = torch.nn.Embedding(num_classes, args.label_embedding_size)
        self.dropout = torch.nn.Dropout(args.dropout)

        self.attention = Attention(args.encoder_hidden_size, args.decoder_hidden_size, args.attention_hidden_size)

        self.rnn = torch.nn.GRU(args.encoder_hidden_size*2 + args.label_embedding_size, args.decoder_hidden_size, args.decoder_num_layers, batch_first=False, bidirectional=False)

        self.hidden2label = torch.nn.Linear(args.decoder_hidden_size, num_classes)

        self.transformer  = torch.nn.Linear(args.encoder_hidden_size*2, args.decoder_hidden_size)
        self.transformer1 = torch.nn.Linear(args.decoder_hidden_size,   args.decoder_hidden_size)
        self.gate = torch.nn.Linear(args.decoder_hidden_size, args.decoder_hidden_size)


    def forward(self, inputs, last_hidden, encoder_outputs, current_encoder_outputs, time_step, max_len, inputs_mask=None):
        embedded = self.label_embedding(inputs).unsqueeze(0)
        embedded = self.dropout(embedded)

        attn_weights = self.attention(last_hidden[-1], encoder_outputs, time_step, max_len, inputs_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        context = context.transpose(0,1)

        rnn_inputs = torch.cat([embedded, context], 2)

        output, hidden = self.rnn(rnn_inputs, last_hidden)
        output = output.squeeze(0)
        
        trans = F.relu(self.transformer(current_encoder_outputs.squeeze(0)))
        trans1= F.relu(self.transformer1(output))
        gate  = self.gate(trans+trans1)
        T = torch.sigmoid(gate)
        C = 1 - T

        output = T*trans + C*output

        output = self.hidden2label(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, args):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, source, source_length, target=None, testing=False):
        source_mask = source > 0
        source = source.transpose(0, 1)
        target_= target
        if target != None:
            target, _ = torch.nn.utils.rnn.pad_packed_sequence(target, total_length=self.args.max_len)
        
        batch_size = source.size(1)
        max_len = source.size(0) #in other sq2seq, max_len should be target.size()
        label_size = self.args.label_size
        outputs =  Variable(torch.zeros(max_len, batch_size, label_size)).cuda()
        attention =Variable(torch.zeros(max_len, batch_size, max_len)).cuda()

        encoder_outputs, hidden = self.encoder(source, source_length)
        hidden = hidden[:self.args.decoder_num_layers]
        output = Variable(torch.zeros((batch_size))).long().cuda()

        for t in range(max_len):
            current_encoder_outputs = encoder_outputs[t,:,:].unsqueeze(0)
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_outputs, current_encoder_outputs, t, max_len, source_mask)
            outputs[t] = output
            attention[t] = attn_weights.squeeze()
            #is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if testing:
                output = Variable(top1).cuda()
            else:
                output = Variable(target[t]).cuda()

        if testing:
            outputs = outputs.transpose(0,1)
            return outputs, attention
        else:
            packed_y = torch.nn.utils.rnn.pack_padded_sequence(outputs, source_length)
            score  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y.data), target_.data)
            return score
