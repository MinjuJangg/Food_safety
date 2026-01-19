import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, BatchEncoding
from .encoder import SumEncoder

Tensor = torch.Tensor

class BertSum_Ext(nn.Module):

    __doc__ = r"""
        Implementation of the paper; 
        https://arxiv.org/pdf/1908.08345.pdf
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
        self.head = SumEncoder(
            enc_num_layers,
            enc_hidden_size,
            enc_intermediate_size,
            enc_num_attention_heads,
            enc_dropout_prob,
        ).eval() # eval()로 설정: dropout 비활성화(사전학습 가정)

        # 이진 분류를 위한 손실함수 (문장이 요약에 포함될지 여부)
        self.loss_fn = nn.BCELoss(reduction='none')


    def forward(
            self,
            encodings: BatchEncoding,
            cls_token_ids: Tensor,
            ext_labels: Optional[Tensor] = None,
    ):  
        token_embeds = self.base_model(**encodings).last_hidden_state
        # cls_logits: 각 문장의 중요도 점수(logits)
        # cls_mask: 실제 문장만 1로 표시된 마스크(패딩 제거)
        _, cls_mask, cls_logits = self.head(token_embeds, cls_token_ids).values()
        scores = cls_logits * cls_mask # 패딩 CLS 무효화

        num_sents = torch.sum(cls_mask, dim=-1)

        loss = None
        if not (self.loss_fn is None or ext_labels is None):
            loss = self.loss_fn(cls_logits, ext_labels.float())
            loss = (loss * cls_mask).sum() / num_sents.sum()

        prediction, confidence = [], []
        for i, score in enumerate(scores):
            conf, pred = torch.sort(score[cls_mask[i] == 1], descending=True, dim=-1)
            prediction.append(pred.tolist()) # 문장 인덱스를 중요도 순으로 나열
            confidence.append(conf) # 중요도 점수

        return {
            'logits': cls_logits,
            'loss': loss,
            'prediction': prediction,
            'confidence': confidence,
        }
