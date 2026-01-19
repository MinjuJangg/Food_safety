import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="config_name", default='bertsum_config.yaml', type=str, help="config file path")
args = parser.parse_args()

@hydra.main(version_base=None, config_path="configs/summarization/bertsum", config_name=args.config_name)
def train(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    model = BertSum_Ext(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    train_df = pd.read_csv(os.path.join(cfg.dataset.path, cfg.dataset.train_file))
    val_df = pd.read_csv(os.path.join(cfg.dataset.path, cfg.dataset.val_file))

    train_dataset = ExtSum_Dataset(train_df, tokenizer, cfg.max_seq_len)
    val_dataset = ExtSum_Dataset(val_df, tokenizer, cfg.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    engine = ExtSum_Engine(model, train_df, val_df, **cfg.engine)
    logger = Another_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()

    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    logger.watch(engine)

    if cfg.train_checkpoint:
        trainer.fit(engine, train_loader, val_loader, ckpt_path=cfg.train_checkpoint)
    else:
        trainer.fit(engine, train_loader, val_loader)

    wandb.finish()

if __name__ == "__main__":
    train()