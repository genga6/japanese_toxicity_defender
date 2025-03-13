import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

from dataset.dataset import ToxicDataset


class CustomTrainer:
    def __init__(
        self, 
        model_name: str = "c1-tohoku/distilbert-base-japanese", 
        num_labels: int = 2, 
        batch_size: int = 8, 
        num_epochs: int = 3, 
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        
        self.metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")

        self.training_args = TrainingArguments(
            output_dir="./results", 
            per_device_train_batch_size=self.batch_size, 
            num_train_epochs=self.num_epochs, 
            evaluation_strategy="epoch", 
            save_strategy="epoch", 
            logging_dir="./logs", 
            logging_steps=10, 
            fp16=True, 
            load_best_model_at_end=True, 
        )

    def compute_metrics(self, eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = self.metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        precision = precision_metric.compute(predictions=preds, references=labels, average="macro")
        recall = recall_metric.compute(predictions=preds, references=labels, average="macro")

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
            "precision": precision["precision"],
            "recall": recall["recall"]
        }

    def train(self, dataset: Dataset) -> None:
        trainer = Trainer(
            model=self.model, 
            args=self.training_args, 
            train_dataset=dataset, 
            compute_metrics=self.compute_metrics, 
        )
        trainer.train()

    def save(self, save_path: str = "model") -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"モデルとトークナイザーを保存しました: {save_path}")