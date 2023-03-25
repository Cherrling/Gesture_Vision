import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import CSLDaily
from model import SLTModel


def main():
    model = SLTModel.load_from_checkpoint(
        "lightning_logs/version_10/checkpoints/epoch=99-step=32300.ckpt",
        map_location="cpu",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=60,
        callbacks=ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
        )
    )
    trainer.test(
        datamodule=CSLDaily(
            train_batch_size=64,
            val_batch_size=64,
            test_batch_size=64,
            num_workers=8,
        ),
        model=model,
    )


if __name__ == "__main__":
    main()
