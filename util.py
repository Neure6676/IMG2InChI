
import pickle
import pandas as pd
import numpy as np
import torch
import pickle
import re
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import cv2
import math
tqdm.pandas(desc='apply')


data_dir = '/Users/zhenyuanzhang/Desktop/udemy/AWS_Lambda_and_Serverless/IMG2InChi/data'

STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

image_size = 224
vocab_size = 193
max_length = 300


class YNakamaTokenizer(object):
    """字符转换类
    """

    def __init__(self, is_load=True):
        self.stoi = {}
        self.itos = {}

        if is_load:
            #{char:index}
            with open(data_dir+'/tokenizer.stoi.pickle','rb') as f:
                self.stoi = pickle.load(f)
            #{index:char}
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        """根据标签生成字典
        """
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        """将str转换成indexlist
        """
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        """将indexlist转换成str
        """
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        """将预测结果(index)转换为字符(char)，组装为标准 InChI 格式
        e.g
        """
        # 添加头部
        inchi = 'InChI=1S/'
        # 遍历预测
        for p in predict:
            # 遇到 <eos> 或 <pad> 中止
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    def text_to_sequence(self, text):
        """将多个str转换成indexlist
        """
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        """将多个indexlist转换成str
        """
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        """将多个预测结果(index)转换为字符(char)，组装为标准 InChI 格式
        """
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi


class FixNumSampler(Sampler):
    """验集和测试集采样
    """
    def __init__(self, dataset, length=-1, is_shuffle=False):
        if length<=0:
            length=len(dataset)

        self.is_shuffle = is_shuffle
        self.length = length


    def __iter__(self):
        index = np.arange(self.length)
        if self.is_shuffle: random.shuffle(index)
        return iter(index)

    def __len__(self):
        return self.length



# def compute_lb_score(predict, truth):
#     """评估函数，编辑距离计算
#     """
#     score = []
#     for p, t in zip(predict, truth):
#         s = Levenshtein.distance(p, t)
#         score.append(s)
#     score = np.array(score)
#     return score


def pad_sequence_to_max_length(sequence, max_length, padding_value):
    """填充文本
    """
    # sequence: batchsize, length
    batch_size =len(sequence)
    # 生成 [batchsize,max_length] 全为 padding_value 的张量
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    # 赋值，起到 padding 作用
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L] = s
    return pad_sequence

def load_tokenizer():
    """加载tokenizer
    """
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    return tokenizer

# from kaggle
def split_form1(form):
    """化学式预处理
    """
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')

def split_form2(form):
    """原子连接预处理
    """
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')

def split_form3(formlst):
    """原子连接预处理
    """
    string = ''
    for form in formlst:
        string+= ' '+split_form2(form)
    return string.rstrip(' ')



def make_train(fold = 6, random_seed=666):
    """生成训练集
    :param: fold 生成的 fold 数
    :param: random_seed 随机种子

    :return: DataFrame  e.g 'image_id', 'InChI', 'formula', 'text', 'sequence', 'length'
    """
    # 加载标记器
    token = load_tokenizer()
    # 读取训练数据 InChI=1S/C13H20OS/c1-9(2)8-15-13-6-5-10(3)7-12(13)11(4)14/h5-7,9,11,14H,8H2,1-4H3
    df_label = pd.read_csv(data_dir+'/bms/train_labels.csv')
    # 加工 formula e.g C13H20OS
    df_label['formula'] = df_label.InChI.progress_apply(lambda x:x.split('/')[1])
    # 加工 text e.g  C 13 H 20 O S /c 1 - 9 ( 2 ) 8 - 15 - 13 - 6 - 5 - 10 ( 3 ) 7 - 12 ( 13 ) 11 ( 4 ) 14 /h 5 - 7 , 9 , 11 , 14 H , 8 H 2 , 1 - 4 H 3
    df_label['text'] = df_label.formula.progress_apply(lambda x:split_form1(x))+df_label.InChI.progress_apply(lambda x:split_form3(x.split('/')[2:]))
    # 将 str 转换成 index
    df_label['sequence'] = df_label.text.progress_apply(lambda x:token.one_text_to_sequence(x))
    # 加工 token 长度
    df_label['length'] = df_label.sequence.progress_apply(lambda x:len(x)-2)
    # 生成 fold 值
    fold_lst = (len(df_label)//fold)*[i for i in range(fold)]
    r=random.random
    random.seed(random_seed)
    random.shuffle(fold_lst,random=r)
    df_label['fold'] = fold_lst
    # 保存为 CSV 数据
    df_label.to_csv(data_dir+'/bms/df_train.csv')


def make_test():
    """生成测试集
    """
    df = pd.read_csv(data_dir+'/bms/sample_submission.csv')
    df_orientation = pd.read_csv(data_dir+'/test_orientation.csv')
    df = df.merge(df_orientation, on='image_id')

    df.loc[:, 'path'] = 'test'
    df.loc[:, 'InChI'] = '0'
    df.loc[:, 'formula'] = '0'
    df.loc[:, 'text']=   '0'
    df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
    df.loc[:, 'length'] = 1

    df_test = df
    df_test.to_csv(data_dir+'/bms/df_test.csv')


#####################################
# 数据加载
#####################################

def rot_augment(r):
    """图像预处理函数
    以 90 度为标准单位对图像进行旋转
    """
    image = r['image']
    h, w = image.shape

    l= r['d'].orientation
    if l == 1:
        image = np.rot90(image, -1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)

    # 图像 resize
    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    return r

def null_augment(r):
    image = r['image']
    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    #print('OK')
    return r


class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, mode = 'train' ,augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.mode = mode
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        if self.mode =='train':
            image_file = r'/content/drive/MyDrive/molecule/server_code/data/bms/{}'.format(self.mode) +'/{}/{}/{}/{}.png'.format(d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        else:
            image_file = r'/content/drive/MyDrive/molecule/server_code/data/bms/{}'.format(self.mode) + '/{}/{}/{}/{}.png'.format(
                d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        #print(image_file)
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite('r_org.jpg', image)
        token = d.sequence
        r = {
            'index': index,
            'image_id': d.image_id,
            'InChI': d.InChI,
            'formula': d.formula,
            'd': d,
            'image': image,
            'token': token,
        }
        if self.mode != 'test':
            #image = self.augment(image=image)['image']
            image = self.augment(r)['image']
            r['image'] = image
        else:
            r = self.augment(r)
        # cv2.imwrite('r_aug.jpg', image)
        #print(image_file)
        return r


def collate_fn(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length: #sort by decreasing length
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)

    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)

    return collate


def np_loss_cross_entropy(probability, truth):
    """np交叉熵损失函数
    :param: array probability [bs,dim]
    :param: array truth       [bs]
    """
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p,1e-6,1))
    loss = loss.mean()
    return loss

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

#############################################
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


#https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionEncode2D(torch.nn.Module):
    def __init__(self, dim, width, height):
        super().__init__()
        assert (dim % 4 == 0)
        self.width  = width
        self.height = height

        dim = dim//2
        d = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        position_w = torch.arange(0., width ).unsqueeze(1)
        position_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim*2, height, width)

        pos[0,      0:dim:2, :, :] = torch.sin(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,      1:dim:2, :, :] = torch.cos(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,dim + 0:   :2, :, :] = torch.sin(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        pos[0,dim + 1:   :2, :, :] = torch.cos(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x + self.pos[:,:,:H,:W]
        return x

# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(torch.nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x
# conda install -y -c rdkit rdkit


if __name__ == '__main__':
    pass