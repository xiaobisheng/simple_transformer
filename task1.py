#!/usr/bin/env python3

# 该脚本实现一个神经网络预测数字序列的任务：
# 给定数字序列XY[0-5]，X和Y是给定的两个数字，计算在剩余数字中X和Y出现次数差（#X - #Y）
#
# 示例:
# 输入: 1213211
# 输出: 2 (3 - 1)
#
# 该脚本使用transformer实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)

SEQ_LEN = 5
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64

# This function generates data samples as described at the beginning of the
# script
def get_data_sample(batch_size=1):
    random_seq = torch.randint(low=0, high=VOCAB_SIZE - 1,
                               size=[batch_size, SEQ_LEN + 2])
    
    gts = torch.zeros(size=[batch_size], dtype=torch.long)
    for i in range(batch_size):
        a = random_seq[i]
        count = torch.bincount(a)
        first_count = count[a[0]]
        second_cound = count[a[1]]
        gts[i] = first_count - second_cound

    gts += SEQ_LEN
    return random_seq, gts

# Network definition
class Net(nn.Module):
    def __init__(self, num_encoding_layers=1, num_hidden=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, num_hidden)
        positional_encoding = torch.empty([SEQ_LEN + 2, 1])
        nn.init.normal_(positional_encoding)
        self.positional_encoding = nn.Parameter(positional_encoding,
                                                requires_grad=True)
        q = torch.empty([1, num_hidden])
        nn.init.normal_(q)
        self.q = nn.Parameter(q, requires_grad=True)
        self.encoding_layers = torch.nn.ModuleList([
                                EncodingLayer(num_hidden, num_heads)
                                for _ in range(num_encoding_layers) ])
        self.decoding_layer = MultiHeadAttention(num_hidden, num_heads)
        self.c1 = nn.Conv1d(num_hidden + 1, num_hidden, 1)
        self.fc1 = nn.Linear(num_hidden, 2 * SEQ_LEN + 1)

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        positional_encoding = self.positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.repeat([B, 1, 1])
        x = torch.cat([x, positional_encoding], axis=-1)
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        q = self.q.unsqueeze(0).repeat([B, 1, 1])
        x = self.decoding_layer(q, x, x)
        x = x.squeeze(1)
        x = self.fc1(x)
        return x

class EncodingLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
        self.c1 = nn.Conv1d(num_hidden, 2 * num_hidden, 1)
        self.c2 = nn.Conv1d(2 * num_hidden, num_hidden, 1)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.norm1(x)
        x1 = x.transpose(1, 2)
        x1 = self.c1(x1)
        x1 = F.relu(x1)
        x1 = self.c2(x1)
        x1 = F.relu(x1)
        x1 = x1.transpose(1, 2)
        x = x + x1
        x = self.norm2(x)
        return x

class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()

    def forward(self, Q, K, V, d_k):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        # 1) QK^T/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores: [batch_size, n_heads, len_q, len_k]
        # 2)  softmax
        attn = nn.Softmax(dim=-1)(scores)  # attn: [batch_size, n_heads, len_q, len_k]
        # 3) muliplied with V
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.n_heads = num_heads
        self.embed_dim = embed_dim
        self.d_k = int(self.embed_dim / self.n_heads)

        self.concat = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # q, k, and v are batch-first
        residual, batch_size = q, q.size(0)
        # 1）linear projection [batch_size, seq_len, d_model] ->  [batch_size, n_heads, seq_len, d_k/d_v]
        Q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # 2）Attention
        context = ScaledDotProductionAttention()(Q, K, V, self.d_k)  # context: [batch_size, n_heads, len_q, d_v]

        # 3）concat部分
        context = torch.cat([context[:, i, :, :] for i in range(context.size(1))], dim=-1)
        result = self.concat(context)

        return nn.LayerNorm(self.embed_dim)(result + residual)

# Instantiate network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# Train the network
for i in range(NUM_TRAINING_STEPS):
    inputs, labels = get_data_sample(BATCH_SIZE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

    if i % 100 == 0:
        print('[%d/%d] loss: %.3f, accuracy: %.3f' %
              (i , NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item()))
    if i == NUM_TRAINING_STEPS - 1:
        print('Final accuracy: %.3f, expected %.3f' %
              (accuracy.item(), 1.0))
