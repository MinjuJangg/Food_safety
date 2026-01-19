import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from .encoder import BertExtEncoder

Tensor = torch.Tensor

class BertExt(nn.Module):
    __doc__ = r"""
        Implementation of the BertExt from the paper; 
        https://arxiv.org/abs/2004.08795
    """
    
    def __init__(
            self,
            base_checkpoint: str,
            enc_num_layers: int = 2, # Number of transformer layers in the encoder
            enc_intermediate_size: int = 2048, # Size of the intermediate layer in the FFN
            enc_num_attention_heads: int = 8, # Number of attention heads in the multi-head attention
            enc_dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.base_checkpoint = base_checkpoint
        self.base_model = AutoModel.from_pretrained(self.base_checkpoint) # 사전학습 모델

        enc_hidden_size = self.base_model.config.hidden_size

        # 문장의 CLS 임베딩들을 BERT 레이어에 통과시켜 score를 계산하는 인코더
        self.head = BertExtEncoder(
            enc_num_layers,
            enc_hidden_size,
            enc_intermediate_size,
            enc_num_attention_heads,
            enc_dropout_prob,
        )

        # 이진 분류를 위한 손실함수 (문장이 요약에 포함될지 여부)
        self.loss_fn = nn.BCELoss(reduction='none')


    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            token_type_ids: Tensor,
            cls_indices: Tensor,
            labels: Optional[Tensor] = None,
    ):  
        
        token_embeds = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state
        
        # HEAD 결과 명시적으로 추출
        head_out = self.head(token_embeds, cls_indices)
        
        if not isinstance(head_out, dict):
            raise ValueError(f'head_out is not a dict: got {type(head_out)}')
        
        cls_embed = head_out.get('cls_embeddings')
        cls_mask = head_out.get('cls_token_mask')
        cls_logits = head_out.get('logits')
        
        if any(v is None for v in [cls_embed, cls_mask, cls_logits]):
            raise ValueError(f"One or more values are missing from head_out: {head_out.keys()}")
        
        if not isinstance(cls_mask, torch.Tensor):
            raise TypeError(f"cls_mask is not a tensor: {type(cls_mask)}")

        # score 계산
        scores = torch.sigmoid(cls_logits).squeeze(-1)
        masked_scores = scores * cls_mask.float()

        loss = None
        if labels is not None:
            loss = self.loss_fn(masked_scores, labels.float())
            loss = (loss * cls_mask).sum() / cls_mask.sum()

        prediction, confidence = [], []
        for i in range(scores.size(0)):
            num_sents = int(cls_mask[i].sum())
            sent_scores = scores[i, :num_sents]
        
            conf, pred = torch.sort(sent_scores, descending=True)
            prediction.append(pred)
            confidence.append(conf)

        return {
            'logits': cls_logits,
            'loss': loss,
            'prediction': prediction,
            'confidence': confidence,
        }