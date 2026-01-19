import os
import datetime
import pandas as pd
import pytorch_lightning as pl
from typing import OrderedDict, Tuple
from torch.optim import AdamW
from .model.bertext import *
from .model.utils import *
from .utils.lr_scheduler import *
from .evaluate.evaluate import RougeScorer


class BertExt_Engine(pl.LightningModule):
    __doc__ = r"""
        pl-based engine for training a BertExt(BertSum without trigram blocking)
        Unlike the english benchmark datasets(CNN/DM etc.), it has human-written extractive labels,
        so we evaluate the model with the given extractive labels instead of the abstractive. 
    
        Args:
            model: model instance to train
            val_df: validation dataset in pd.DataFrame
            test_df: test dataset in pd.DataFrame

            lr: learning rate
            betas: betas of torch.optim.Adam
            weight_decay: weight_decay of torch.optim.Adam
            adam_epsilon: eps of torch.optim.Adam
            sum_size: # sentences in a model-predicted summary
            save_result: save test result
             
        train_df, val_df and test_df must be given in order to get the candidate summary 
        from the prediction by indexing the document.
    """
    
    def __init__(
        self,
        model,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,

        lr: float = None,
        betas: Tuple[float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        sum_size: int = 3,
        save_result: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'val_df', 'test_df'])
        
        self.model = model
        self.val_df = val_df
        self.test_df = test_df
        
        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight'] # weight decay 없이 학습
        
        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optim_params, self.hparams.lr, betas=self.hparams.betas, eps=self.hparams.adam_epsilon)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }
          
    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            cls_indices=batch['cls_indices'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, logger=True, on_epoch=True)
        self._train_outputs.append(loss)
        
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            cls_indices=batch['cls_indices'],
            labels=batch['labels']
        )
        loss = outputs['loss']
        preds = outputs['prediction']
        
        self.log('val_loss_step', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        ref_sums, can_sums = [], []
        accs = []
        for i, id in enumerate(batch['id']):
            sample = self.val_df[self.val_df['id'] == id].squeeze()
            text = sample['text']
            
            ref_sum = [text[i] for i in sample['extractive']]
            ref_sums.append('\n'.join(ref_sum))
            
            can_sum = get_candidate_sum(text, preds[i], self.sum_size)
            can_sums.append('\n'.join(can_sum))
            
            # accuracy 계산
            ref_indices = set(sample['extractive'])
            pred_indices = set(preds[i][:self.sum_size])
            
            if len(ref_indices) > 0:
                acc = len(ref_indices & pred_indices) / len(ref_indices)
            else:
                acc = 0.0
            accs.append(acc)
        
        output = {
            'loss': loss,
            'ref_sums': ref_sums,
            'can_sums': can_sums,
            'accs': accs
        }
        
        return output

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bat=True)

        all_ref_sums = [item for output in outputs for item in output['ref_sums']]
        all_can_sums = [item for output in outputs for item in output['can_sums']]
        all_accs = [item for output in outputs for item in output['accs']]
        
        r1, r2, rL = 0, 0, 0
        for ref, can in zip(all_ref_sums, all_can_sums):
            score = self.scorer.score(ref, can)
            r1 += score['rouge1'].fmeasure
            r2 += score['rouge2'].fmeasure
            rL += score['rougeL'].fmeasure
        
        num_samples = len(all_ref_sums)
        if num_samples > 0:
            r1 /= num_samples
            r2 /= num_samples
            rL /= num_samples
            acc = sum(all_accs) / len(all_accs)
        else:
            acc = 0
        
        self.log_dict({
            'val_rouge1': r1 * 100,
            'val_rouge2': r2 * 100,
            'val_rougeL': rL * 100,
            'val_acc': acc * 100,
        }, prog_bar=True, logger=True)
    

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            cls_indices=batch['cls_indices'],
        )
        preds = outputs['prediction']

        # 평가 로직
        texts, ref_sums, can_sums = [], [], []
        ref_indices_list, can_indices_list, accs = [], [], []

        for i, doc_id in enumerate(batch['id']):
            sample = self.test_df.loc[doc_id.item()] 
            sentences = sample['text']
            texts.append('\n'.join(sentences))

            ref_indices = sample['extractive']
            ref_sum = '\n'.join([sentences[idx] for idx in ref_indices])
            ref_sums.append(ref_sum)
            ref_indices_list.append(ref_indices)

            pred_indices_tensor = preds[i]
            top_indices = pred_indices_tensor.topk(self.hparams.sum_size).indices.tolist()
            can_sum = '\n'.join([sentences[idx] for idx in sorted(top_indices)])
            can_sums.append(can_sum)
            can_indices_list.append(sorted(top_indices))

            ref_set = set(ref_indices)
            pred_set = set(top_indices)
            if len(ref_set) > 0:
                acc = len(ref_set & pred_set) / len(ref_set)
                accs.append(acc)

        return {
            'texts': texts,
            'ref_sums': ref_sums,
            'can_sums': can_sums,
            'ref_indices': ref_indices_list,
            'can_indices': can_indices_list,
            'accs': accs
        }


    def on_test_epoch_end(self, outputs):
        result = {
            'text': [],
            'reference_summary': [],
            'candidate_summary': [],
            'reference_indices': [],
            'candidate_indices': []
        }
        r1, r2, rL, all_accs = [], [], [], []

        print('Calculating ROUGE Score & ACC for the test set...')
        for output in outputs:
            for ref, can in zip(output['ref_sums'], output['can_sums']):
                score = self.scorer.score(ref, can)
                r1.append(score['rouge1'].fmeasure)
                r2.append(score['rouge2'].fmeasure)
                rL.append(score['rougeL'].fmeasure)
            
            all_accs.extend(output['accs'])
            
            if self.hparams.save_result:
                result['text'].extend(output['texts'])
                result['reference_summary'].extend(output['ref_sums'])
                result['candidate_summary'].extend(output['can_sums'])
                result['reference_indices'].extend(output['ref_indices'])
                result['candidate_indices'].extend(output['can_indices'])
        
        num_samples = len(r1)
        rouge1 = (sum(r1) / num_samples) * 100 if num_samples > 0 else 0
        rouge2 = (sum(r2) / num_samples) * 100 if num_samples > 0 else 0
        rougeL = (sum(rL) / num_samples) * 100 if num_samples > 0 else 0
        accuracy = (sum(all_accs) / len(all_accs)) * 100 if all_accs else 0

        self.log_dict({
            'test_rouge1': rouge1,
            'test_rouge2': rouge2,
            'test_rougeL': rougeL,
            'test_acc': accuracy
        })
        print(f"Test ROUGE-1: {rouge1:.2f}, ROUGE-2: {rouge2:.2f}, ROUGE-L: {rougeL:.2f}, ACC: {accuracy:.2f}")

        if self.hparams.save_result:
            path = f'./result/{datetime.datetime.now().strftime("%y-%m-%d")}'
            if not os.path.exists(path):
                os.makedirs(path)
            
            result_df = pd.DataFrame(result)
            result_df.to_csv(f'{path}/{datetime.datetime.now().strftime("%H-%M-%S")}_test_results.csv', index=False)
            print(f"Test results saved to {path}")