from datasets import load_dataset, load_metric

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import nltk
import numpy as np

from .upload import upload

def train(
    raw_datasets,
    model_checkpoint,
    tokenizer,
    model,
    data_collator,
    metric
):
    max_input_length = 1024
    max_target_length = 128

    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # 레이블에 해당하는 부분은 as_target_tokenizer로 토크나이징 함.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        # 레이블과 input데이터를 병합
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


    batch_size = 2
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-xsum", # output dir
        evaluation_strategy = "epoch", # 각 epoch 마다 evaluate
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3, # 용량을 줄이기 위해서 이전 체크 포인트를 삭제(최대 3개 저장)
        num_train_epochs=1, # epochs
        predict_with_generate=True, # (ROUGE, BLEU)메트릭을 생성할 지 여부.
        fp16=True, # 16 bit training
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics 
    )

    trainer.train()

if __name__ == "__main__":
    nltk.download('punkt')
    raw_datasets = load_dataset("xsum")
    metric = load_metric("rouge")

    # 토크나이저 불러오기
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    train(
        raw_datasets=raw_datasets,
        model_checkpoint=model_checkpoint,
        tokenizer=tokenizer,
        model=model,
        data_collator=data_collator,
        metric=metric
    )
    
    upload()
