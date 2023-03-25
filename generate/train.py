import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import CSLDaily
from model import SLTModel


def main():
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2000,
        callbacks=ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
        )
    )
    trainer.fit(
        datamodule=CSLDaily(
            train_batch_size=64,
            val_batch_size=64,
            test_batch_size=64,
            num_workers=8,
        ),
        model=SLTModel(
            vocab_size=2000 + 106,
            learning_rate=1e-3,
            max_epochs=2000,
        ),
    )


if __name__ == "__main__":
    main()
