import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import kss

class BertExt_Dataset(Dataset):
    """
    Custom Dataset for Food Safety CSV File
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        method: str,
        max_len: int=512, 
        label_model: SentenceTransformer=None, 
        sim_threshold=0.6
        ):
        """
        Args:
            df (pd.DataFrame): DataFrame which contains 'extracted', 'summary' column
            tokenizer (BertTokenizer): Tokenizer for Bert Input
            max_len (int) : max input length for Sentence Transformer
            label_model (SentenceTransformer): Sentence Embedding Model for label generation
            sim_threshold (float) : cosine similarity threshold for label generation
        """
        
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.method = method
        
        self.label_model = label_model
        self.sim_threshold = sim_threshold
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        source_text = str(row['내용'])
        target_text = str(row[f'{self.method}'])
        
        # 1. separate extracted text by sentences
        source_sentences = kss.split_sentences(source_text)
        
        # 2. Label generation by LaBSE Semantic Similarity
        labels = torch.zeros(len(source_sentences), dtype=torch.float)
        if self.label_model and source_sentences and target_text:
            source_embeddings = self.label_model.encode(source_sentences, convert_to_tensor=True)
            target_embedding = self.label_model.encode(target_text, convert_to_tensor=True)
            
            similarities = cos_sim(source_embeddings, target_embedding)
            labels = (similarities.squeeze(1) >= self.sim_threshold).float()
        
        # 3. Tokenizing as BERT input form
        src_subtokens = []
        segments_ids = []
        cls_indices = []
        
        src_subtokens.append(self.tokenizer.cls_token)
        segments_ids.append(0)
        
        src_subtokens = []
        for i, sent in enumerate(source_sentences):
            tokens = self.tokenizer.tokenize(sent)
            tokens = tokens[:self.max_len - 2]
            
            if len(src_subtokens) + len(tokens) + 2 > self.max_len:
                break
            
            cls_indices.append(len(src_subtokens))
            
            segment_id = i % 2
            
            src_subtokens.append(self.tokenizer.cls_token)
            src_subtokens.extend(tokens)
            src_subtokens.append(self.tokenizer.sep_token)
            
            segments_ids.extend([segment_id] * (len(tokens) + 2))
        
        labels = labels[:len(cls_indices)]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        attention_mask = [1] * len(input_ids)
        
        return {
            'id': row.get('id', -1),
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(segments_ids),
            'cls_indices': torch.tensor(cls_indices, dtype=torch.long),
            'labels': labels
        }
        
def collate_fn(batch, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    
    max_input_len = max(item['input_ids'].shape[0] for item in batch)
    max_cls_len = max(item['cls_indices'].shape[0] for item in batch)

    collated_batch = {
        'input_ids': torch.full((len(batch), max_input_len), pad_token_id, dtype=torch.long),
        'attention_mask': torch.zeros((len(batch), max_input_len), dtype=torch.long),
        'token_type_ids': torch.zeros((len(batch), max_input_len), dtype=torch.long),
        'cls_indices': torch.full((len(batch), max_cls_len), -1, dtype=torch.long),
        'labels': torch.full((len(batch), max_cls_len), -1.0, dtype=torch.float)
    }
    
    for i, item in enumerate(batch):
        input_len = len(item['input_ids'])
        collated_batch['input_ids'][i, :input_len] = item['input_ids']
        collated_batch['attention_mask'][i, :input_len] = 1
        collated_batch['token_type_ids'][i, :input_len] = item['token_type_ids']
        
        cls_len = len(item['cls_indices'])
        collated_batch['cls_indices'][i, :cls_len] = item['cls_indices']
        collated_batch['labels'][i: cls_len] = item['labels']
    
    return collated_batch