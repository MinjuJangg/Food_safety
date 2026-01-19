import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding

class ExtSum_Dataset(Dataset):

    __doc__ = r"""
        Referred to the repos below;
        https://github.com/nlpyang/PreSumm
        https://github.com/KPFBERT/kpfbertsum

        Returns:
            ids: 'id' value of the data, which is to index the document from the prediction
            encodings: input_ids, token_type_ids, attention_mask
                token_type_ids alternates between 0 and 1 to separate the sentences
            cls_token_ids: identify CLS tokens representing sentences among all tokens
            ext_label: extractive label to train in sentence-level binary classification
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad = tokenizer.pad_token_id # padding token의 id


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # load and tokenize each sentence
        encodings = []
        for sent in row['method2_texts']: # raw_text_split
            encoding = self.tokenizer(
                sent,
                add_special_tokens=True,
                truncation=True,
                max_length=512
            )
            encodings.append(encoding)

        input_ids, token_type_ids, attention_mask = [], [], []
        cls_token_ids = []
        #ext_label = []
        

        # seperate each of sequences
        seq_id = 0 # for segment embedding
        for enc in encodings:
            if seq_id > 1:
                seq_id = 0
            
            current_length = len(input_ids)
            new_tokens = len(enc['input_ids'])
            
            
            # 길이 제한 체크 (추가하기 전에)
            if current_length + new_tokens > self.max_seq_len:
                break
            
            cls_token_ids += [len(input_ids)] # each [CLS] symbol collects features for the sentence preceding it.
            input_ids += enc['input_ids']
            token_type_ids += len(enc['input_ids']) * [seq_id]
            attention_mask += len(enc['input_ids']) * [1]
            
            seq_id += 1
        
        # pad inputs
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += pad_len * [self.pad]
            token_type_ids += pad_len * [0]
            attention_mask += pad_len * [0]

        # adjust for BertSum_Ext
        # 모델에 입력으로 넣기 위해 길이를 통일시킴
        if len(cls_token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(cls_token_ids)
            cls_token_ids += pad_len * [-1]
            #ext_label += pad_len * [0]

        encodings = BatchEncoding(
            {
                'input_ids': torch.tensor(input_ids),
                'token_type_ids': torch.tensor(token_type_ids),
                'attention_mask': torch.tensor(attention_mask),
            }
        )
            
        return_data = dict(
            id=row.name,
            encodings=encodings,
            cls_token_ids=torch.tensor(cls_token_ids),
            cat1 = str(row['대분류']),
            cat2 = str(row['중분류']),
            cat3 = str(row['소분류']),
        )
        
        return return_data