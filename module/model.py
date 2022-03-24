import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):#基础的神经网络模型
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, n_head: int, ff_dim: int, embed_drop: float, hidden_drop: float):
        super().__init__()
        #embedding一个低维的向量表示一个单词
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)#保存字典并用来检索，输入为带下标的列表
        # num_embeddings(int) - 嵌入字典的大小
        # embedding_dim(int) - 每个嵌入向量的大小
        # padding_idx(int, optional) - 如果提供的话，输出遇到此下标时用零填充
        # max_norm(float, optional) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
        # norm_type(float, optional) - 对于max_norm选项计算p范数时的p
        # scale_grad_by_freq(boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=ff_dim, dropout=hidden_drop)
        # 多头自注意力回馈网络
        # 下面的encoder为layer的实例，num_layers – the number of sub-encoder-layers in the encoder (required).

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)#对输入进行线性变换 y = xA^T + b

        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x, mask):
        x = x.transpose(0, 1)#转置
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        return x

    def forward(self, x, *args):
        # (batch_size, max_seq_len, embed_dim)
        mask = args[0] if len(args) > 0 else None
        tok_emb = self.tok_embedding(x)
        max_seq_len = x.shape[-1]#取列
        pos_emb = self.pos_embedding(torch.arange(max_seq_len).to(x.device))
        x = tok_emb + pos_emb.unsqueeze(0)#在第0维增加一个维度
        x = self.embed_dropout(x)
        x = self.linear1(x)
        x = self.encode(x, mask)
        x = self.linear2(x)
        probs = torch.matmul(x, self.tok_embedding.weight.t())#矩阵Tensor乘法
        return probs


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=n_layer,
                              dropout=rnn_drop if n_layer > 1 else 0, batch_first=True, bidirectional=True)
        #将一个多层的 (LSTM) 应用到输入序列。input_size – 输入的特征维度；hidden_size – 隐状态的特征维度；num_layers – 层数（和时序展开要区分开；
        #bias – 如果为False，那么LSTM将不会使用，默认为TRUE；dropout – 如果非零的话，将会在RNN的输出上加个dropout，最后一层除外；
        # bidirectional – 如果为True，将会变成一个双向RNN，默认为False（BILSTM）双向长短时记忆网络
        self.embed_dropout = nn.Dropout(embed_drop)
        #随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的（p=0.5)。
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x, _ = self.bilstm(x)
        return x

    def predict(self, x):
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        #nn.Embedding.weight随机初始化方式是标准正态分布 N(0,1)
        return probs

    def forward(self, x, *args):
        x = self.encode(x)
        return self.predict(x)


class BiLSTMAttn(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        x = x.transpose(0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        return self.predict(x)


class BiLSTMCNN(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        #一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out），in_channels(int) – 输入信号的通道，out_channels(int) – 卷积产生的通道
        #kerner_size(int or tuple) - 卷积核的尺寸；stride(int or tuple, optional) - 卷积步长；padding (int or tuple, optional)- 输入的每一条边补充0的层数

    def forward(self, x, *args):
        x = self.encode(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        return self.predict(x)


class BiLSTMConvAttRes(BiLSTM):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        #normalized_shape： 输入尺寸
# [∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]
# eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
# elementwise_affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数。
#归一化层：BN，LN，IN，GN从学术化上解释差异：
# BatchNorm：batch方向做归一化，算NHW的均值，对小batchsize效果不好；BN主要缺点是对batchsize的大小比较敏感，
    # 由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布
# LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；
# InstanceNorm：一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，
    # 所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
# GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束。
# SwitchableNorm是将BN、LN、IN结合，赋予权重，让网络自己去学习归一化层应该使用什么方法。

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        res = x
        x = self.conv(x.transpose(1, 2)).relu()
        x = x.permute(2, 0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        x = self.norm(res + x)
        return self.predict(x)


class CNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, embed_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs
