#import区域
from collections import OrderedDict
import torch
import torch.distributed
import torch.multiprocessing.spawn
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import os
import functools
import random
import time

from typing import Union, Optional
import datasets
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast, BertForQuestionAnswering, Blip2Model, T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer
import tqdm
from datasets import load_dataset
import torch.distributed as dist
import torch.utils.data._utils.collate


#实用工具函数
#1.计算函数，用来计算损失losses
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


#模型函数
#1.定义多层感知机 Mlp (Multi layer perceptron),也就是多层的前馈神经网络  是一种全连接网络 
#embed_dim 是输入的数据的维度  也就是词向量的维度为256
#mlp_ratio是mlp的比率，比如我们给入一个维度为64的输入向量 那么mlp的第一层输出向量维度为256 = 64*4 ，作为第二层输入，第二层输入256维度 输出64=256/4维度
#bias 是偏置 神经网络里需要有偏置神经元bias = True 是将偏置神经元加入了考量
#layer_act act是activate的缩写 layer_act是层与层之间的激活函数，使用的是pytorch库里的nn库，内部自己写好的GELU函数
#定义一个神经网络 我们最重要的就是两点：定义这个神经网络怎么初始化(__init__)、定义这个神经网络怎么前向传播
class Mlp(nn.Module):
    def __init__(self, dim = 256, mlp_ratio = 4.0, bias = True, layer_activate = nn.GELU, dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        #第一个零件是embed_dim输入转成mlp_dim的输出 是线性层1  比如输入256维的向量 设置mlp_ratio==4.0 那么输出就是1024维度的向量
        self.fully_connected_layer1 = nn.Linear(dim, mlp_dim, bias = bias, **factory_kwargs)
        #第二个零件是激活层，使用GELU函数对数据加入非线性，因为第一层是nn.Linear是完全线性的,我们需要引入激活层对整个结构加入非线性
        self.activate = layer_activate()
        #第三个零件是线性层2 把mlp_dim的输入转成和这个神经网络的输入一样的维度embed_dim
        self.fully_connected_layer2 = nn.Linear(mlp_dim, dim, bias = bias, **factory_kwargs)
        #第四个零件是随机丢弃 模拟多次训练的情形 这边设置了dropout参数为0.0即随机丢失的概率为0.0 不会丢失
        self.drop = nn.Dropout(dropout)

    def forward(self, input:torch.tensor):
        result_fully_connected_layer1 = self.fully_connected_layer1(input)
        result_activate = self.activate(result_fully_connected_layer1)
        result_activate_drop = self.drop(result_activate)
        result_fully_connected_layer2 = self.fully_connected_layer2(result_activate_drop)
        result = self.drop(result_fully_connected_layer2)
        return result

#2.自注意力机制
#这里我们需要注意一个词叫projection——投影 我们在后面就能看到它的作用
#qkv_bias = False意味着我们在引入Linear层的时候，只有矩阵乘法
class selfattention(nn.Module):
    def __init__(self, dim, num_heads = 16, qkv_bias = False, attention_drop = 0.0, out_proj_drop = 0.0, device = None, dtype = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        #第一个零件是一个线性层叫输入投影层  我们输入一个维度为dim的向量  输出一个维度为3 * dim的向量  这其实就是使用embedding后的词向量去得到K Q V矩阵
        #也对应着投影 把dim维度的向量投影成 3 * dim维度的向量
        self.input_projection_layer = nn.Linear(dim, 3 * dim, bias = qkv_bias, **factory_kwargs)
        #第二个零件是一个线性层叫输出投影层 我们输入维度为dim的向量  输出一个维度为dim的向量  这其实就是softmax(QK^T)V = A V中模拟A乘以V的过程
        self.output_projection_layer = nn.Linear(dim, dim, bias = qkv_bias, **factory_kwargs)
        #第三个零件是attention层的drop out
        self.attention_drop = nn.Dropout(attention_drop)
        #第四个零件是输出层的drop out
        self.out_drop = nn.Dropout(out_proj_drop)

        #使用我们自己写的初始化函数把模型数据都掷成初始状态
        self._reset_parameters()

    def _reset_parameters(self):
        #把输入层的权重reset
        nn.init.xavier_uniform_(self.input_projection_layer.weight)
        #把输出层的权重reset
        nn.init.xavier_uniform_(self.output_projection_layer.weight)
        
        if self.input_projection_layer.bias is not None:
            nn.init.constant_(self.input_projection_layer.bias, 0.0)
        if self.output_projection_layer.bias is not None:
            nn.init.constant_(self.output_projection_layer.bias, 0.0)

    #前向传播
    #attention_mask = None即此处不适用掩盖的attention机制
    #is_causal = False即此处不使用因果的attention机制，也就是mask
    #scale = None也就是不使用因子去控制注意力机制分数，在Transformer里面这个因子其实是  1/sqrt(d_k)
    def forward(self, input, scale = None, attention_mask = None, is_causal = False):
        #Batch Length Channel举个实例
        #输入I love you 这是一句话 Batch == 1,也就是只有一句话（batch也就是一次处理的数据多少）   Length == 3 这句话里面有3个词    Channel == 256
        #Channel == 256 是embedding的时候，把一个词变成一个256的token（token是基本单元的意思）  所以生成的矩阵input就会是256*3的大小，并且是个张量(tensor)
        #如果我们访问input的shape，会返回(Batch,Length,Channel)这样的返回值
        Batch, Length, Channel = input.shape
        #使用输入投影层把输入维度为dim的向量投影成维度为3*dim的向量
        #再使用torch包内自带的分割函数chunk  对self.input_projection_layer层的最后一个维度3 * dim这个维度进行分割
        #分割成3个维度  把3 * dim分割成dim
        q, k, v = torch.chunk(self.input_projection_layer(input), chunks = 3, dim = -1)
        q = q.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)#(B,H,L,C)
        k = k.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)
        v = v.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)

        #step1 先判断有没有scale传进来
        if scale is None:
            scale = 1.0 / math.sqrt(Channel // self.num_heads)

        #step2 使用scale控制     Q @ K^T
        #q (B,H,L,C)
        attention_score_matrix = scale * (q @ k.transpose(-2,-1))#(B,H,L,L)

        #step3 看是否因果，如果是因果，但是非掩码 那就把mask矩阵作成一个下三角的
        if is_causal is True and (attention_mask is None):
            attention_mask = torch.tril(q.new_ones((Length, Length)))
        
        #step4 如果是掩码的，那就把那些因果里为0的元素变成负无穷(因为我们用的是softmax)
        if attention_mask is not None:
            assert attention_mask.shape == (Length,Length)
            attention_score_matrix = attention_score_matrix.masked_fill(attention_mask == 0, float('-inf'))

        #沿着最后那个维度做Softmax
        attention_score_matrix = F.softmax(attention_score_matrix, dim = -1)
        attention_score_matrix = self.attention_drop(attention_score_matrix)
        #attention_score_matrix (B,H,L,L)  v(B,H,L,C)
        #attention_score_matrix @ v (B,H,L,C) ————> (B,L,H,C)
        store = (attention_score_matrix @ v).transpose(1,2).contiguous().view(Batch, Length, Channel)
        output = self.out_drop(self.output_projection_layer(store))
        return output
    
#3.将注意力机制selfattention和多层感知机Mlp拼装成一个小块 block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4.0, qkv_bias = False, drop = 0.0, attention_drop = 0.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm, Attn = selfattention):
        super().__init__()
        self.normalization1 = norm_layer(dim)
        #def __init__(self, dim, num_heads = 16, qkv_bias = False, attention_drop = 0.0, out_proj_drop = 0.0, device = None, dtype = None)
        #需要注意的是这里做了实例化
        self.attention = Attn(dim, num_heads = num_heads, qkv_bias = qkv_bias, attention_drop = attention_drop, out_proj_drop = drop)
        #drop_path层暂时没有任何操作
        self.drop_path = nn.Identity()
        self.normalization2 = norm_layer(dim)
        #def __init__(self, embed_dim=256, mlp_ratio = 4.0, bias = True, layer_activate = nn.GELU, dropout=0.0, device=None, dtype=None)
        self.mlp = Mlp(dim, mlp_ratio = mlp_ratio, layer_activate = act_layer, dropout = drop)

    #normalization1 ----->  selfattention ----->  add&norm  ----->  mlp  ----->add
    def forward(self, input):
        result_normalization1 = self.normalization1(input)
        #因为self.attention已经做了实例化
        #所以self.attention(store, is_causal)等价于self.attention.forward(store, is_causal) 
        result_attention = self.attention(result_normalization1, is_causal = True)
        result_drop1 = self.drop_path(result_attention)
        #Add
        store = result_drop1 + input
        #Norm
        result_normalization2 = self.normalization2(store)
        #同理和上面attention处相同，这里self.mlp同样也是做了实例化了，我们只需要把store给到forward即可
        result_mlp = self.mlp(result_normalization2)
        result_drop2 = self.drop_path(result_mlp)
        #Add
        result = result_drop2 + store
        return result
    #我们只通过SelfAttention实现，所以不使用断言代码  师姐的断言代码里面包括了selflinearattention  selflstm

#4.分词器 对文章、单词做预处理
class Tokenizer:
    #静态方法load_tokenizer，可以无需实例化调用
    #这个静态方法加载网上已经训练好了的分词器
    @staticmethod
    def load_tokenizer(tokenizer_path = "/home/ChenYufan/gpt/tokenizer", max_sequence_length = 512):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.model_max_length = max_sequence_length
        vocab_size = tokenizer.vocab_size
        model_max_length = tokenizer.model_max_length

        #GPT2 分词器是没有pad_token的开始填充标识的
        #所以我们需要把句子开始填充的标识等于结束eos_tokenizer的标识
        if (hasattr(tokenizer, "pad_token") is False) or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        #BertTokenizer 和 BertTokenizerFast
        #不进行小写处理，所以如果训练好的分词器对输入的句子进行小写处理
        #我们就关闭小写处理
        if hasattr(tokenizer, "do_lower_case") is True:
            tokenizer.do_lower_case = False

        #tokens_to_ids默认输入为字符串列表，可以是list也可以是tuple
        def tokens_to_ids(text = [""]):
            #断言text文档是一个列表或者是一个元组
            assert isinstance(text, list or tuple)
            #断言文档的第一个元素是字符串
            #也就是说你至少要有输入
            assert isinstance(text[0], str)
            #输出需要是pytorch中的Tensor模式 return_tensors = 'pt'
            #允许进行填充padding
            #如果输入文本太长，允许进行截断分割truncation
            ids = tokenizer(text, return_tensors = 'pt',padding = True, truncation = True)
            #返回输入的id，也就是把输入的词变成一个一个的id数字
            #并且返回输入对应的mask也就是掩码，这个指示我们什么地方填充padding了，什么地方没有
            return ids.input_ids, ids.attention_mask
        
        def ids_to_tokens(input_ids):
            #将input_ids 也就是输入对应的id序列，进行批量解码batch_decode
            #跳过特殊的token，可能是在进行编码的时候加入到了文本中的
            return tokenizer.batch_decode(input_ids, skip_special_tokens = True)
        
        return tokenizer, vocab_size, model_max_length, tokens_to_ids, ids_to_tokens
    
#5.数据集
class Dataset:
    @staticmethod
    #Linux    /tmp/tmp.bin
    def data_store(dset:datasets.Dataset, total_batches = 1024, save = r"binary"):
        arr_len = np.sum(dset['len'], dtype = np.uint64)
        dtype = np.int32
        os.makedirs(os.path.dirname(save), exist_ok = True)
        arr = np.memmap(save, dtype = dtype, mode = 'w+', shape = (arr_len,))
        
        idx = 0
        #for in tqdm.tqdm(range,desc)本身是和for in range没区别的，只是加入了进度条
        for batch_idx in tqdm.tqdm(range(total_batches), desc = f'writing to {save}'):
            batch = dset.shard(num_shards = total_batches, index = batch_idx, contiguous = True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        #刷新内存 保证写入磁盘
        arr.flush()

    @classmethod
    def load_tinystories(cls, path = "/home/ChenYufan/gpt/txt", num_proc = 64, tokenizer_path = "/home/ChenYufan/gpt/tokenizer", save = r"/home/ChenYufan/gpt/bin"):
        dataset_train = datasets.load_dataset(path=path,  data_files=["TinyStoriesV2-GPT4-train.txt"], num_proc=16)
        dataset_val = datasets.load_dataset(path=path,  data_files=["TinyStoriesV2-GPT4-valid.txt"], num_proc=16)
        split_dataset = datasets.DatasetDict(train = dataset_train['train'], val = dataset_val['train'])
        print(split_dataset)
        print("\n")
        print(split_dataset['train'])
        print("\n")
        print(split_dataset['val'])
        print("\n")
        tokenized = split_dataset
        
        if tokenizer_path is not None:
            tokenizer = Tokenizer.load_tokenizer(tokenizer_path = tokenizer_path)[0]
            def process(example):
                ids = tokenizer(example["text"]).input_ids
                out ={'ids':ids, 'len':len(ids)}
                return out
        
            tokenized = split_dataset.map(process, remove_columns=['text'], desc = "the program is now tokenizing the split_dataset", num_proc = num_proc)
            print(f"using tokenizer {tokenizer_path}", flush = True)
        
        if save is not None:
            assert isinstance(tokenized, datasets.DatasetDict)
            cls.data_store(tokenized["train"], save = f"{save.rstrip('/')}/train.bin")
            cls.data_store(tokenized["val"], save = f"{save.rstrip('/')}/val.bin")
        return tokenized
    

    @classmethod
    def build_dataloader(cls, dataset: str = None, batch_per_gpu:int=32, tokens_to_ids = None, max_seqlen: int = 512, num_workers: int = 16, max_iters: int = 10000, shuffle = True, raw_text = False, distributed = True):
        #这里要在正式训练的时候才知道为什么要看分布式训练的进程多少，这里还不理解
        if distributed is True:
            num_gpus = dist.get_world_size()
        else:
            num_gpus = torch.cuda.device_count()
        
        # 原文类型的数据集 DS_raw
        # 封装处理文本数据集
        # max_iter * batchsize = total_number
        # max_iter 是告诉我们 一个epoch需要往设备里输入多少次数据，这就和一次输入的数据量密切相关
        # 所以max_iter就是这个批次里面到底有多少个数据，就是len
        class DS_Raw(torch.utils.data.Dataset):
            def __init__(self, dataset: datasets.Dataset, max_iter = None, max_seqlen = max_seqlen, tokens_to_ids = tokens_to_ids):
                self.dataset = dataset
                if max_iter is None:
                    self.max_iter = len(dataset)
                else:
                    self.max_iter = max_iter

                self.tokens_to_ids = tokens_to_ids
                self.max_seqlen = max_seqlen
            
            def __len__(self):
                return self.max_iter
            
            def __getitem__ (self, idx):
                # 保证我们去取一个样例的时候，idx怎么输入都不超过实际的大小
                idx = idx % len(self.dataset)
                # 在 dataset 里面取 text 的操作
                text = self.dataset[idx]['text']
                # input_ids 是 tokenizer作用后得到的张量
                # attention_masks 是 tokenizer作用后返回的时候进行了 padding的检测标志
                # 这里做了Truncation = True了，为什么还要这样写？
                input_ids, attention_masks = self.tokens_to_ids([text])
                inputs_tragets: torch.Tensor = input_ids[:, :self.max_seqlen + 1]
                masks = (attention_masks[:, :self.max_seqlen + 1])[:, :-1].contiguous()
                return inputs_tragets[0], masks[0]
        
            
        class DS_Map(torch.utils.data.Dataset):
            def __init__(self, dataset: str, max_iter=None, max_seqlen=max_seqlen, drop_last=False):
                self.seqlen = max_seqlen
                self.samples = np.memmap(dataset, dtype=np.int32, mode='r')
                self.len_samples = len(self.samples)
                if drop_last:
                    self.sample_div_seqlen = self.len_samples // self.seqlen
                else:
                    self.sample_div_seqlen = math.ceil(self.len_samples / self.seqlen)
                self.max_iter = self.sample_div_seqlen if max_iter is None else max_iter
        
            def __len__(self):
                return self.max_iter

            def __getitem__(self, idx):
                idx = idx % self.sample_div_seqlen
                if idx * self.seqlen + self.seqlen + 1 > self.len_samples:
                    to_pad = idx * self.seqlen + self.seqlen - self.len_samples
                    inputs_targets = torch.cat([torch.from_numpy(self.samples[idx * self.seqlen:].astype(np.int64)), torch.from_numpy(self.samples[:to_pad + 1].astype(np.int64))])
                    """print(inputs_targets)
                    breakpoint()"""
                else:
                    inputs_targets = torch.from_numpy(self.samples[idx * self.seqlen: idx * self.seqlen + self.seqlen + 1].astype(np.int64))
                    """print(inputs_targets)
                    breakpoint()"""
                masks = torch.ones((self.seqlen))
                return inputs_targets, masks

        collate_fn_map: dict = torch.utils.data._utils.collate.default_collate_fn_map
        collate_fn_map.update({
            torch.Tensor: lambda batch, collate_fn_map: torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        }) 
        collate_fn = functools.partial(torch.utils.data._utils.collate.collate, collate_fn_map=collate_fn_map) 


        dataset = (DS_Raw if raw_text else DS_Map)(dataset=dataset, max_iter=max_iters * batch_per_gpu * num_gpus)
        
        if (shuffle is True) and (distributed is not True):
            _shuffle = True
        else:
            _shuffle = False

        if distributed is False:
            _sampler = None
        else:
            _sampler = torch.utils.data.DistributedSampler(dataset = dataset, shuffle = shuffle, drop_last = False)
        

        data_loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batch_per_gpu,
            shuffle = _shuffle,
            sampler = _sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            pin_memory = True,
        )

        return data_loader

#6.搭建GPT本体
#nlayers是排列、搭建的block的多少，block是normalization、transformer、add一体的一个小块
class toyGPT(nn.Module):
    def __init__(self, max_seqlen = 512, blocklayers = 16, embed_dim = 256, num_heads = 16, drop = 0.0, tokenizer_path = "/home/ChenYufan/gpt/tokenizer"):
        super().__init__()
        # return tokenizer, vocab_size, model_max_length, tokens_to_ids, ids_to_tokens
        self.vocab_size = Tokenizer.load_tokenizer(tokenizer_path = tokenizer_path, max_sequence_length = max_seqlen)[1]
        self.tokens_to_ids = Tokenizer.load_tokenizer(tokenizer_path = tokenizer_path, max_sequence_length = max_seqlen)[3]
        self.ids_to_tokens = Tokenizer.load_tokenizer(tokenizer_path = tokenizer_path, max_sequence_length = max_seqlen)[4]
        self.max_seqlen = max_seqlen

        #arg1:size of dictionary of embedding
        #arg2:size of each embedding vector
        self.layer_token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.layer_position_embedding = nn.Embedding(max_seqlen, embed_dim)
        if drop == 0.0:
            self.token_drop = nn.Identity() 
        else:
            self.token_drop = nn.Dropout(drop)

        #顺序性进行组合排列，把我们写好的类似block、embedding都排在一起
        #transformer的输入是经过了词向量embedding和位置embedding的数据
        self.transformer = nn.Sequential(*[Block(dim = embed_dim, num_heads = num_heads, drop = drop) for _ in range(blocklayers)])
        #normalization层并不改变数据维度，只对数据做0均值化、方差归一化
        self.normalization = nn.LayerNorm(embed_dim, bias = False)
        self.lm_head = nn.Linear(embed_dim, self.vocab_size, bias = False)

        weight_typing = True
        if weight_typing:
            # https://paperswithcode.com/method/weight-tying
            self.layer_token_embedding.weight = self.lm_head.weight

        #total_params用于获取整个模型内的所有参数
        #self.named_parameters是toyGPT继承自nn.Module的一个方法   它作用于self本身用于获取self之中所有的命名好的参数
        #self.named_parameters返回的是(name,parameter)
        #k遍历所有name
        #v遍历所有的参数对应张量
        total_params = sum({k: v.numel() for k, v in self.named_parameters()}.values())
        transformer_params = sum({k: v.numel() for k, v in self.transformer.named_parameters()}.values())

        print(f"model info: vocab_size:{self.vocab_size}, max_seqlen:{max_seqlen}, total_params:{total_params}, transformer_params:{transformer_params}")

    #targets是用来计算losses的
    #我们如果要计算losses就要保证输入和输出的shape是一样的
    #recurrent没加
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None):
        #inputs_ids是使用tokenizer处理过后的
        #B——batch size是输入句子数目或者说一次拿多少数据做计算     L——length是有多少词     F——feature也就是embedding_dim
        #经过tokenizer，输出形如:
        
        #input_ids = torch.tensor([
        #[101, 2054, 2003, 1037, 2338, 1012, 102, 0, 0, 0],  # 第一个样本的输入序列
        #[101, 1045, 2031, 1037, 2338, 1012, 102, 2045, 2055, 2003]  # 第二个样本的输入序列
        #input_ids.shape = (2,10)
        #也就是还没有做到F维度，(B,L)到(B,L,C)是embedding里面做的
        
        B, L = input_ids.shape#(B,L)
        if targets is not None:
            assert input_ids.shape == targets.shape
            if masks is not None:
                assert input_ids.shape == masks.shape

        #使用embedding，(B,L)变成(B,L,C)
        embds = self.layer_token_embedding(input_ids)#(B,L,C)
        #这里的位置编码是可学习的(learnable) 不使用sin   cos那套  详情见《attention is all you need》
        position_embeds = self.layer_position_embedding.weight
        if L != self.max_seqlen:
            assert L < self.max_seqlen
            position_embeds = self.layer_position_embedding.weight[:L]
        
        embedded = embds + position_embeds
        embedded = self.token_drop(embedded)
        
        embedded = self.transformer(embedded)
        
        #配合最后那个孤立的add 最后一个add & norm模块
        embedded = self.normalization(embedded)

        if targets is not None:
            logits = self.lm_head(embedded)
            #把(Batch_size, Length, embedding_dim)重构成(Batch_size * Length, vocab_size)的样子
            #Batch_size * Length = num_vocab，也就是所有的词
            #logits.view(-1, logits.size(-1))即最后一个维度不变，另外的两个维度怎么合成成为一个维度自行计算
            #logits ————> (batch, length, vocab_size)
            #target ————> (bacth * length)  target还是数字编码，也就是每一位的label
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction = "none")
            if masks is not None:
                loss = (loss * masks.view(-1)).sum()/masks.sum()
            else:
                loss = loss.mean()
        else:
            #没有targets，直接输出预测logits，输出loss时使用None
            #没有targets的情况就是我们只有第一位
            assert B == 1
            logits = self.lm_head(embedded[:,[-1],:])
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt = "", max_new_tokens = 512, temperature = 0.7, top_k = None, eos_token_id = 50256):
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype 

        input_ids, attn_mask = self.tokens_to_ids([prompt])#输入的形状为(B,L)
        print("input_ids:",input_ids,'\n')
        print("attn_mask:",attn_mask,'\n')
        #breakpoint()
        input_ids = input_ids.to(dtype = torch.long, device = device)
        output_ids = input_ids.clone()
        """print("output_ids:",output_ids,'\n')
        breakpoint()"""

        # 最多生成 512 个新词
        for _ in range(max_new_tokens):
            logits, loss = self.forward(input_ids = input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # top_k < logits.size(-1)也就是小于vocab_size，那就取前k个
                # top_k > logits.size(-1)那就取全部，不过这个情况不可能出现
                v = torch.topk(logits, min(top_k, logits.size(-1)))
                # 我们只在最高概率的k个里面进行选，这里赋 -infty 后续做softmax时可以得到概率为 0
                logits[logits < v[:, [-1]]] = - float('inf')
            
            # 沿着最后一个dim做softmax， 也就是沿着vocab_size那个维度做softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # 依据概率进行抽取，选择只抽取一个 返回值是抽取的索引
            idx_next = torch.multinomial(probs, num_samples=1)
            print('idx_next:',idx_next,'\n')
            #breakpoint()
            # 把输出放在input_ids后面
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            # 把输出放到输出里面
            output_ids = torch.cat((output_ids, idx_next), dim=1)
            # 如果这里我们生成了 end of text，就结束生成
            if idx_next == eos_token_id:
                break
        
        # 只需要返回的output中
        output = self.ids_to_tokens(output_ids)[0]
        """print("output:",output)
        breakpoint()"""
        return output
    
    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True, assign: bool = False):
        if all([k.startswith("module.") for k in state_dict.keys()]):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict.update({k[len("module."):]: v})
            state_dict = new_state_dict
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

def train(
    # rank,
    batch_per_gpu=32, 
    gradient_accumulation=1, 
    resume=None, 
    dataset_path="/home/ChenYufan/gpt/txt", 
    dataset_bin_path = "/home/ChenYufan/gpt/bin/train.bin",
    tokenizer_path="/home/ChenYufan/gpt/tokenizer", 
    model_path="/home/ChenYufan/gpt/model_transformer",
    raw_text=False,
):
    seed = 42
    distributed = False
    enable_autocast = False # MUST BE FALSE!
    grad_clip = 1.0
    max_iters = 12000
    base_lr = 1e-3
    max_seqlen = 512
    print_interval = 10 
    save_iter_interval = 1000
    shuffle = True
    num_workers = 16
    
    
    if False:
        os.environ['MASTER_PORT'] = '12345'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['RANK'] = str(rank)
        print(torch.cuda.device_count())

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(f'cuda:{dist.get_rank()}')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    num_gpus = dist.get_world_size() if distributed else 1
    is_master_proc = (not distributed) or (dist.get_rank() == 0)

    batch_size = batch_per_gpu * gradient_accumulation * num_gpus
    lr = base_lr / 256 * batch_size
    if is_master_proc:
        print(f"batch_size: {batch_per_gpu} * {num_gpus} * {gradient_accumulation} = {batch_size}.", flush=True)
        print(f"lr: {base_lr} / {256} * {batch_size} = {lr}.", flush=True)

    model = toyGPT(tokenizer_path=tokenizer_path, max_seqlen=max_seqlen).cuda()
    model = nn.parallel.DistributedDataParallel(model) if distributed else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1000),
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000),
    ], milestones=[2 * max_iters // 3])
    # lr_scheduler =  torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #     torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=1000),
    #     torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters - 1000, eta_min=0.1 * lr),
    # ], milestones=[1000])

    tokens_to_ids = model.module.tokens_to_ids if distributed else model.tokens_to_ids

    #dataset_train_txt = Dataset.load_tinystories(path = dataset_path)['train']
    dataset_train_bin = dataset_bin_path

    dataloader_train = Dataset.build_dataloader(
        dataset=dataset_train_bin,
        batch_per_gpu=batch_per_gpu, 
        tokens_to_ids=tokens_to_ids,
        max_seqlen=max_seqlen,
        num_workers=num_workers,
        max_iters=max_iters * gradient_accumulation,
        shuffle=shuffle,
        raw_text=raw_text,
        distributed=distributed,
    )

    start_iter = 0
    if resume is not None:
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_iter = start_iter['iter_num']

    losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_per_iter = 0
    optimizer.zero_grad()
    time_start = time.time()
    model.train()
    print(model)
    with open("/home/ChenYufan/gpt/model_transformer/model.txt",'w')as f:
        f.write(str(model))


    for it, data in enumerate(dataloader_train):
        it = it + start_iter # resume
        if it == max_iters * batch_per_gpu * gradient_accumulation:
            break

        input_targets, masks = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
        inputs, targets = input_targets[:, :-1].contiguous(), input_targets[:, 1:].contiguous()

        inputs = inputs.long()
        targets = targets.long()

        time_data = time.time() - time_start

        do_gradient_accumulate = (it + 1) % gradient_accumulation == 0

        # we do not need to sync gradient if optimizer.step is not needed
        model.require_backward_grad_sync = do_gradient_accumulate
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enable_autocast):
            loss = model(inputs, targets, masks)[1]
            # print(masks[0].sum())
            loss = loss / gradient_accumulation
            loss.backward()
            loss_per_iter += loss.item()

        if do_gradient_accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()
            time_batch = time.time() - time_start

            if is_master_proc:
                real_it = (it + 1) // gradient_accumulation
                os.makedirs(model_path, exist_ok=True)
                save_file = f"{model_path.rstrip('/')}/model_iter{real_it}.pth"

                losses.update(loss_per_iter)
                data_time.update(time_data * gradient_accumulation)
                batch_time.update(time_batch * gradient_accumulation)

                if (real_it + 1) % print_interval == 0:
                    print(f"iter [{real_it}/{max_iters}] "
                        f"loss {losses.val:.4f}({losses.avg:.4f}) "
                        f"lr {optimizer.param_groups[0]['lr']:.8g} "
                        f"data_time {data_time.val:.4f}({data_time.avg:.4f})s "
                        f"batch_time {batch_time.val:.4f}({batch_time.avg:.4f})s "
                        f"eta [{batch_time.avg * (max_iters - real_it) / 60:.0f}/{batch_time.avg * max_iters / 60:.0f}] min ", flush=True)
                    
                if (real_it + 1) % save_iter_interval == 0 or real_it == 99:
                    print(f"saving to {save_file}", flush=True)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': it,
                    }, save_file)
                    
            if distributed:
                dist.barrier()
            loss_per_iter = 0
        time_start = time.time()


if __name__ == "__main__":
    if False:
        from functools import partial
        _train = partial(train, batch_per_gpu= 32, gradient_accumulation=16, raw_text=False, dataset_path="/home/ChenYufan/gpt/txt", dataset_bin_path = "/home/ChenYufan/gpt/bin/train.bin")
        os.environ['MASTER_PORT'] = '12345'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['WORLD_SIZE'] = '4'
        print(torch.cuda.device_count())
        torch.multiprocessing.spawn(_train, (), nprocs=4, start_method="spawn")
        
    train(batch_per_gpu= 64, gradient_accumulation=2, raw_text=False, dataset_path= "/home/ChenYufan/gpt/txt", dataset_bin_path = "/home/ChenYufan/gpt/bin/train.bin")
        
        








