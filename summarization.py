from transformers import pipeline
from preprocess import *

def predict(url, max_length=120, min_length=30):
    # todo: 한국어 기능 추가
    title, article = text_extraction(url)
    sentences = text_preprocessing(article)
    chunks = make_chunks(sentences)

    # todo: 모델 저장하고 다시 불러오는 코드로 실행시간 단축하기
    summarizer = pipeline("summarization")
    pred = summarizer(chunks, max_length=120, min_length=30, do_sample=False)

    summary = " "
    for sub_text in pred:
        summary += sub_text['summary_text']    

    print(f"results: {summary}")

    return summary

if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()
    
    summary = predict(args.url)
    title, article = text_extraction(args.url)
    print(f"before summarization: {len(summary)}")
    print(f"after summarization: {len(article)}")