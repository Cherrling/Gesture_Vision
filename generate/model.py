import json

import torch
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer, AutoModel, BertConfig, EncoderDecoderConfig, EncoderDecoderModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput


class SLTModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int = 2000 + 106,
        learning_rate: float = 1e-3,
        max_epochs: int = 60,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=BertConfig.from_pretrained("bert-base-chinese"),
            decoder_config=BertConfig(
                vocab_size=vocab_size,
                hidden_size=128,
                num_hidden_layers=6,
                num_attention_heads=4,
                intermediate_size=512,
            )
        )
        self.model = EncoderDecoderModel(
            config=config,
            encoder=AutoModel.from_pretrained("bert-base-chinese"),
        )
        # self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        #     "bert-base-chinese", "bert-base-chinese"
        # )

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id

        for param in self.model.encoder.parameters():
            param.requires_grad_(False)

        with open("data/CSL-Daily/gloss_vocabs.json", "r") as f:
            gloss_vocabs = json.load(f)
        self.token_to_gloss = {
            i: gloss for gloss, i in gloss_vocabs.items()
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        self.model.encoder.eval()
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
    ) -> torch.no_grad():
        return self.model.generate(input_ids)

    def training_step(self, batch, batch_idx: int):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs.loss

        preds = outputs.logits.argmax(-1)
        for preds_i, labels_i in zip(preds, labels):
            preds_i = [self.token_to_gloss[i] for i in preds_i.tolist() if i in self.token_to_gloss]
            labels_i = [self.token_to_gloss[i] for i in labels_i.tolist() if i in self.token_to_gloss]
            print(preds_i, labels_i)

        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def main():
    model = SLTModel()


if __name__ == "__main__":
    main()
