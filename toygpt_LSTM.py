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



class Mlp(nn.Module):
    def __init__(self, dim = 256, mlp_ratio = 4.0, bias = True, layer_activate = nn.GELU, dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.fully_connected_layer1 = nn.Linear(dim, mlp_dim, bias = bias, **factory_kwargs)
        self.activate = layer_activate()
        self.fully_connected_layer2 = nn.Linear(mlp_dim, dim, bias = bias, **factory_kwargs)
        self.drop = nn.Dropout(dropout)

    def forward(self, input:torch.tensor):
        result_fully_connected_layer1 = self.fully_connected_layer1(input)
        result_activate = self.activate(result_fully_connected_layer1)
        result_activate_drop = self.drop(result_activate)
        result_fully_connected_layer2 = self.fully_connected_layer2(result_activate_drop)
        result = self.drop(result_fully_connected_layer2)
        return result


class selfattention(nn.Module):
    def __init__(self, dim, num_heads = 16, qkv_bias = False, attention_drop = 0.0, out_proj_drop = 0.0, device = None, dtype = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.embed_dim = dim
        self.num_heads = num_heads
        self.input_projection_layer = nn.Linear(dim, 3 * dim, bias = qkv_bias, **factory_kwargs)
        self.output_projection_layer = nn.Linear(dim, dim, bias = qkv_bias, **factory_kwargs)
        self.attention_drop = nn.Dropout(attention_drop)
        self.out_drop = nn.Dropout(out_proj_drop)
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


    def forward(self, input, scale = None, attention_mask = None, is_causal = False):

        Batch, Length, Channel = input.shape
        q, k, v = torch.chunk(self.input_projection_layer(input), chunks = 3, dim = -1)
        q = q.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)#(B,H,L,C)
        k = k.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)
        v = v.contiguous().view(Batch, Length, self.num_heads, Channel // self.num_heads).transpose(1,2)


        if scale is None:
            scale = 1.0 / math.sqrt(Channel // self.num_heads)


        #q (B,H,L,C)
        attention_score_matrix = scale * (q @ k.transpose(-2,-1))#(B,H,L,L)


        if is_causal is True and (attention_mask is None):
            attention_mask = torch.tril(q.new_ones((Length, Length)))
        
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


# hidden_size 是 h_t c_t的dimension
class selflstm(nn.Module):
   def init(self, dim:int, proj_drop=0.0, num_layer=16, qbv_bias=False ,attn_drop=0.0, device=None, dtype=None, ):
    factory_kwargs = {'device':device, 'dtype':dtype}
    super().__init__()
    self.LSTM = nn.LSTMCell(input_size=dim, hidden_size=dim)
    self.out_dropout = nn.Dropout(proj_drop)

    def forward(self, input, scale=None,**kwargs):
        #(B, L, C)
        B, L, C = input.shape
        x = input.permute(1, 0, 2)#(L, B, C)

        out = []
        start = True
        # c:cell state    h:hidden state
        for i in range(x.shape[0]):
            # x[i].shape = (B, C)
            hx, cx = self.LSTM(x[i], None if start else (hx, cx))
            start = False
            out.append(hx)
        out = self.out_dropout(torch.stack(out,dim=1))
        return out




"""class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4.0, qkv_bias = False, drop = 0.0, attention_drop = 0.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm, Attn = selfattention):
        super().__init__()
        self.normalization1 = norm_layer(dim)
        #def __init__(self, dim, num_heads = 16, qkv_bias = False, attention_drop = 0.0, out_proj_drop = 0.0, device = None, dtype = None)
        self.attention = Attn(dim, num_heads = num_heads, qkv_bias = qkv_bias, attention_drop = attention_drop, out_proj_drop = drop)
        self.drop_path = nn.Identity()
        self.normalization2 = norm_layer(dim)
        #def __init__(self, embed_dim=256, mlp_ratio = 4.0, bias = True, layer_activate = nn.GELU, dropout=0.0, device=None, dtype=None)
        self.mlp = Mlp(dim, mlp_ratio = mlp_ratio, layer_activate = act_layer, dropout = drop)

    #normalization1 ----->  selfattention ----->  add&norm  ----->  mlp  ----->add
    def forward(self, input):
        result_normalization1 = self.normalization1(input)
        #self.attention(store, is_causal)等价于self.attention.forward(store, is_causal) 
        result_attention = self.attention(result_normalization1, is_causal = True)
        result_drop1 = self.drop_path(result_attention)
        #Add
        store = result_drop1 + input
        #Norm
        result_normalization2 = self.normalization2(store)
        result_mlp = self.mlp(result_normalization2)
        result_drop2 = self.drop_path(result_mlp)
        #Add
        result = result_drop2 + store
        return result
    #我们只通过SelfAttention实现，所以不使用断言代码  师姐的断言代码里面包括了selflinearattention  selflstm"""


class Tokenizer:
    @staticmethod
    def load_tokenizer(tokenizer_path = "/home/ChenYufan/gpt/tokenizer", max_sequence_length = 512):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.model_max_length = max_sequence_length
        vocab_size = tokenizer.vocab_size
        model_max_length = tokenizer.model_max_length

        if (hasattr(tokenizer, "pad_token") is False) or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if hasattr(tokenizer, "do_lower_case") is True:
            tokenizer.do_lower_case = False

        def tokens_to_ids(text = [""]):
            assert isinstance(text, list or tuple)
            assert isinstance(text[0], str)
            ids = tokenizer(text, return_tensors = 'pt',padding = True, truncation = True)
            return ids.input_ids, ids.attention_mask
        
        def ids_to_tokens(input_ids):
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
        if distributed is True:
            num_gpus = dist.get_world_size()
        else:
            num_gpus = torch.cuda.device_count()

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
                idx = idx % len(self.dataset)
                text = self.dataset[idx]['text']
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


class toyGPT(nn.Module):
    def __init__(self, max_seqlen = 512, blocklayers = 6, embed_dim = 256, drop = 0.0, tokenizer_path = "/home/ChenYufan/gpt/tokenizer"):
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


        self.transformer = nn.LSTM(input_size=embed_dim ,hidden_size=embed_dim, num_layers=blocklayers, dropout=drop, batch_first=True)

        self.normalization = nn.LayerNorm(embed_dim, bias = False)
        self.lm_head = nn.Linear(embed_dim, self.vocab_size, bias = False)

        weight_typing = True
        if weight_typing:
            # https://paperswithcode.com/method/weight-tying
            self.layer_token_embedding.weight = self.lm_head.weight

        total_params = sum({k: v.numel() for k, v in self.named_parameters()}.values())
        transformer_params = sum({k: v.numel() for k, v in self.transformer.named_parameters()}.values())

        print(f"model info: vocab_size:{self.vocab_size}, max_seqlen:{max_seqlen}, total_params:{total_params}, transformer_params:{transformer_params}")

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
        
        # for nn.LSTM
        if isinstance(embedded, tuple):
            embedded = embedded[0]

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
            assert B == 1
            logits = self.lm_head(embedded[:,[-1],:])
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt = "", max_new_tokens = 512, temperature = 0.2, top_k = None, eos_token_id = 50256):
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
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #breakpoint()
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            output_ids = torch.cat((output_ids, idx_next), dim=1)
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
    model_path="/home/ChenYufan/gpt/model_LSTM",
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
        
        








