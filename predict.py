from transformers import (T5ForConditionalGeneration, 
                          T5TokenizerFast as T5Tokenizer,
                          pipeline)

from preprocess import *

class Summarizer():
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        self.pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

    def predict(self, url, chunk_len=3000):
        title, text = get_title_and_text(url)

        num_iters = int(len(text)/chunk_len)
        summarized_text = []

        for i in range(0, num_iters + 1):
            start = 0
            start = i * chunk_len
            end = (i + 1) * chunk_len
            try:
                out = self.pipeline(text[start:end])
                out = out[0]
                out = out['summary_text']
                summarized_text.append(out)
            except:
                pass

        return summarized_text

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