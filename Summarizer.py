from transformers import (T5ForConditionalGeneration, 
                          T5TokenizerFast as T5Tokenizer,
                          pipeline)
from pororo import Pororo
from utils import *

class Summarizer():
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, language):
        self.ko_pipeline = Pororo(task="summarization", model="abstractive", lang="ko")
        print("한국어 모델 로드 완료")
    
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.en_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("영어 모델 로드 완료")

    def predict(self, url):
        title, text = get_title_and_text(url)
        if title == "":
            return ""

        # 만약 텍스트가 영어면 
        if language == "en":
            self.pipeline = self.en_pipeline
        else:
            self.pipeline = self.ko_pipeline

        short_summary = ""
        long_summary = ""
        try:
            short_summary = text2chunk_and_pred(text, self.pipeline, 64, 16)
            if len(text) > 10000:
                print("텍스트가 너무 길어, 요약을 한번 더 진행합니다.")
                long_summary = text2chunk_and_pred(summary, self.pipeline, 128, 32)
            return short_summary, long_summary

        except:
            return ""


if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()
    
    summarizer = Summarizer("t5-large")
    summarizer.get_model()
    summary = summarizer.predict(args.url)

    print(f"results: {summary}")