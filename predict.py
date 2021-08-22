from transformers import pipeline
from preprocess import *

def summarizer(url, chunk_len=3000):
    title, text = get_title_and_text(url)

    summarizer = pipeline('summarization')

    num_iters = int(len(text)/chunk_len)
    summarized_text = []

    for i in range(0, num_iters + 1):
        start = 0
        start = i * chunk_len
        end = (i + 1) * chunk_len
        try:
            out = summarizer(text[start:end])
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
    
    summary = summarizer(args.url)

    print(f"results: {summary}")

    # python summarization.py --url "https://medium.com/tensorflow/using-tensorflow-2-for-state-of-the-art-natural-language-processing-102445cda54a"