from transformers import (T5ForConditionalGeneration, 
                          T5TokenizerFast as T5Tokenizer,
                          pipeline)

from utils import *

class Summarizer():
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        self.pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("success load model")

    def predict(self, url):
        title, text = get_title_and_text(url)
        if title == "":
            return ""

        try:
            summary = text2chunk_and_pred(text, self.pipeline, 64, 16)
            if len(text) > 10000:
                print("텍스트가 너무 길어, 요약을 한번 더 진행합니다.")
                summary = text2chunk_and_pred(summary, self.pipeline, 128, 32)
            return summary

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

    # python summarization.py --url "https://medium.com/tensorflow/using-tensorflow-2-for-state-of-the-art-natural-language-processing-102445cda54a"