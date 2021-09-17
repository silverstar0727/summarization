from transformers import (T5ForConditionalGeneration, 
                          T5TokenizerFast as T5Tokenizer,
                          pipeline)
from pororo import Pororo
from utils import *

class Summarizer():
    """
    summarization을 얻을 수 있습니다.

    Args: 
        - en_model_name(String) = "t5-large"
    
    Usage:
        url = "https:// ~~~"
        summarizer = Summarizer("t5-large")
        short_summ, long_summ = summarizer.predict(url)
    """
    
    def __init__(self, en_model_name="t5-large"):
        self.en_model_name = en_model_name

        # 한국에 모델을 로드합니다.
        self.ko_pipeline = Pororo(task="summarization", model="abstractive", lang="ko")
        print("한국어 모델 로드 완료")
    
        # 영어 모델을 로드합니다.
        model = T5ForConditionalGeneration.from_pretrained(self.en_model_name)
        tokenizer = T5Tokenizer.from_pretrained(self.en_model_name)
        self.en_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("영어 모델 로드 완료")

    def predict(self, url):
        # 텍스트를 얻습니다.
        # (이것보다 한국어를 판단하기에 나은코드가 있으려나... 같은 일을 두번하게 되는 부분)
        text = get_text(url)
        if text == "":
            return "", ""

        # 텍스트의 종류에 따라서 한국어와 영어 모델을 선택합니다.
        if isKorean(text) == 0:
            self.pipeline = self.en_pipeline
        else:
            self.pipeline = self.ko_pipeline

        short_summary = ""
        long_summary = ""
        try:
            short_summary = text2chunk_and_pred(text, self.pipeline, 64, 16)
            if len(text) > 10000:
                print("텍스트가 너무 길어, 요약을 한번 더 진행합니다.")
                long_summary = text2chunk_and_pred(short_summary, self.pipeline, 128, 32)
            return short_summary, long_summary

        except:
            return short_summary, long_summary


if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()
    
    summarizer = Summarizer("t5-large")
    short_summ, long_summ = summarizer.predict(args.url)

    print(f"results")
    print(short_summ)
    print(long_summ)