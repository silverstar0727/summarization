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
        if title == "":
            return ""

        sentences = text_preprocessing(text)
        chunks = make_chunks(sentences)

        try:
            res = self.pipeline(chunks, max_length=50, min_lenth=0)

            summarized_text = ""
            for i in res:
                summarized_text += res[0]["summary_text"]
                
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