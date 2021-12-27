from datasets import load_dataset, load_metric

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import nltk
import numpy as np


nltk.download('punkt')
raw_datasets = load_dataset("xsum")
metric = load_metric("rouge")

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(raw_datasets["train"])

# 토크나이저 불러오기
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# 토크나이저에 대한 테스트
print(tokenizer("Hello, this one sentence!"))
print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
with tokenizer.as_target_tokenizer():
    print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
    

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


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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




import os
import glob
from google.cloud import storage

bucket = os.environ.get("BUCKET") #(1)
working_dir = os.environ.get("WORKING_DIR") #(2)

# gcs에 디렉토리 업로드
def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path, json_credential_path):
    storage_client = storage.Client.from_service_account_json(json_credential_path) # gcp에서 json 인증파일을 받아와야합니다.
    bucket = storage_client.get_bucket(bucket_name)
    
    assert os.path.isdir(local_path)
    
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, 
                                          bucket, 
                                          gcs_path + "/" + os.path.basename(local_file),
                                          json_credential_path)
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            print(bucket)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


# gcs에 model directory 내용을 업로드
upload_local_directory_to_gcs(local_path="./model", 
                              bucket_name=bucket, 
                              gcs_path=working_dir, 
                              json_credential_path="creds.json")#(3)
