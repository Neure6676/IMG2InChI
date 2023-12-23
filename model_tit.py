from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from tit import *
from util import Swish,PositionEncode2D,PositionEncode1D
from transformer import *

from torch import Tensor
from typing import Optional
import torch
import numpy as np

image_dim   = 384
text_dim    = 384
decoder_dim = 384
num_layer = 3
num_head  = 8
ff_dim = 1024

STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}


image_size = 224
vocab_size = 193
max_length = 300 


class Encoder(torch.nn.Module):
    """编码器类
    这里是使用 tit 作为编码器
    """
    def __init__(self,):
        super(Encoder, self).__init__()
        # 生成网络加载预训练权重  #这里就相当于TNT模块
        self.e = tnt_s_patch16_224(pretrained=False)
        #self.e = tnt_s_patch16_224(pretrained=True)

    
    def forward(self, image):
        """前向传播
        """
        batch_size, C, H, W = image.shape
        x = 2 * image - 1
        # 像素 embeding
        pixel_embed = self.e.pixel_embed(x, self.e.pixel_pos)

        # patch embeding
        patch_embed = self.e.norm2_proj(self.e.proj(self.e.norm1_proj(pixel_embed.reshape(batch_size, self.e.num_patches, -1))))
        patch_embed = torch.cat((self.e.cls_token.expand(batch_size, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.e.patch_pos
        patch_embed = self.e.pos_drop(patch_embed)

        # 多个模块
        for blk in self.e.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        # layer norm
        patch_embed = self.e.norm(patch_embed) #torch.Size([7, 197, 384])

        x = patch_embed
        return x

class Net(torch.nn.Module):
    """编码器+解码器
    """
    def __init__(self,):
        super(Net, self).__init__()
        #  编码器
        self.encoder = Encoder()
        self.image_encode = torch.nn.Identity()

        # 解码器
        self.text_pos    = PositionEncode1D(text_dim,max_length)
        self.token_embed = torch.nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        # logit层
        self.logit  = torch.nn.Linear(decoder_dim, vocab_size)
        self.dropout = torch.nn.Dropout(p=0.5)

        # 初始化参数
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    @torch.jit.unused
    def forward(self, image, token, length):
        """训练阶段：前向传播
        """
        device = image.device
        batch_size = len(image)

        # 
        image_embed = self.encoder(image)

        # 
        #image_embed = self.image_encode(image_embed)
        image_embed = self.image_encode(image_embed).permute(1,0,2) 
        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        text_mask = np.triu(np.ones((max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        #
        x = self.text_decode(text_embed, image_embed, text_mask)
        x = x.permute(1,0,2).contiguous()

        logit = self.logit(x)
        return logit

    @torch.jit.export
    def forward_argmax_decode(self, image):
        """预测阶段：前向传播
        """
        #device = image.device
        batch_size = len(image)

        #同上
        image_embed = self.encoder(image)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        # b*n 填充 <pad>
        #token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long).to(device)
        token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long)
        # 获取输出向量的位置向量
        text_pos = self.text_pos.pos
        # 第一个设置为 <sos>
        token[:,0] = STOI['<sos>']
        eos = STOI['<eos>']
        pad = STOI['<pad>']
        # fast version
        # https://github.com/pytorch/fairseq/blob/21b8fb5cb1a773d0fdc09a28203fe328c4d2b94b/fairseq/sequence_generator.py#L245-L247
        if 1:
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            for t in range(max_length-1):
                # 目前 token 的最后一个值
                last_token = token[:, t]
                # 向量嵌入
                text_embed = self.token_embed(last_token)
                # 加上位置向量
                text_embed = text_embed + text_pos[:,t] #
                # b*text_dim -> 1*b*text_dim
                text_embed = text_embed.reshape(1,batch_size,text_dim)
                # 得到下一个向量 1*b*text_dim
                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                # b*text_dim
                x = x.reshape(batch_size,decoder_dim)
                # b*vocab_size
                l = self.logit(x)
                # 以最大的作为预测
                k = torch.argmax(l, -1)
                token[:, t+1] = k
                # 遇到 <eos> 和 <pad> 停止预测
                if ((k == eos) | (k == pad)).all():  break
        # 返回除了 <sos> 之外的序列
        predict = token[:, 1:]
        return predict

# 损失函数
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss


# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_anti_focal_cross_entropy_loss(logit, token, length):
    gamma = 0.5 # {0.5,1.0}
    label_smooth = 0.90

    # b*seq
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    # b*seq*dim <pad>
    # 压紧序列 x*dim
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    # 压紧序列 x
    truth = pack_padded_sequence(truth, L, batch_first=True).data

    # x*dim
    logp = F.log_softmax(logit, -1)
    # 把logp 上对应的真实标签的 logp 取出来 (ce 公式)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    # output x
    # logp -> p
    p = logp.exp()
    # 加权重，p越大权重越大，和 focal 是反着的
    loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return loss


# check #################################################################

def run_check_net():
    batch_size = 7
    C,H,W = 3, 224, 224
    image = torch.randn((batch_size,C,H,W))

    token  = np.full((batch_size, max_length), STOI['<pad>'], np.int64) #token
    length = np.random.randint(5,max_length-2, batch_size)
    length = np.sort(length)[::-1].copy()
    for b in range(batch_size):
        l = length[b]
        t = np.random.choice(vocab_size,l)
        t = np.insert(t,0,     STOI['<sos>'])
        t = np.insert(t,len(t),STOI['<eos>'])
        L = len(t)
        token[b,:L]=t

    token  = torch.from_numpy(token).long()

    #---
    net = Net()
    net.train()

    logit = net(image, token, length)
    print(logit.shape)
    print(token.shape)
    loss = seq_anti_focal_cross_entropy_loss(logit, token, length)

    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')
    print(length)
    print(length.shape)
    print(token.shape)
    print(image.shape)
    print('---')

    print(logit.shape)
    print(loss)
    print('---')

    net.eval()
    predict = net.forward_argmax_decode(image)
    print(predict.shape)



# main #################################################################
if __name__ == '__main__':
     run_check_net()