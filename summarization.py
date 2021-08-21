from transformers import pipeline
from preprocess import *

def predict(url, max_length=120, min_length=30):
    title, article = text_extraction(url)
    sentences = text_preprocessing(article)
    chunks = make_chunks(sentences)

    summarizer = pipeline("summarization")
    pred = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    print(f"results: {pred}")

    return pred

if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()
    
    pred = predict(args.url)