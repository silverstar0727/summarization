from bs4 import BeautifulSoup
import requests 
from youtube_transcript_api import YouTubeTranscriptApi
import transformers
import re

# input: youtube url -> output: transcript text
def video_extractor(url):
    video_id = url.split("=")[1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
    except:
        return ""

    text = ""
    for i in range(0, len(transcript), 1):
        if i % 2 == 1:
            text += '. ' + transcript[i]['text']
        else:
            text += ' ' + transcript[i]['text']

    return text

# input: post url -> output: post text
def post_extractor(url):
    #todo:  medium, github(제목 이슈, 마크다운 수식 이슈), velog, tistory(제목이슈), brunch(공백이슈)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    results = soup.find_all(['p'])

    # results = soup.find_all(['h1', 'p])
    text = [result.text for result in results]
    text = ' '.join(text)

    return text

# get text from url
def get_text(url):
    if url[:24] == "https://www.youtube.com/":
        text = video_extractor(url)
    elif ("https://medium.com/" in url) or ("tistory.com" in url) or ("https://brunch.co.kr/" in url):
        text = post_extractor(url)
    else:
        text = ""
    
    return text

def text_preprocessing(text):
    # todo: 데이터 전처리 케이스별 정리
    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>')
    text = text.replace("\xa0", "")
    text = text.replace("\n", "")
    sentences = text.split('<eos>')

    return sentences

# 메모리 제한을 피하기 위해서 chunk로 분할
def make_chunks(sentences, max_chunk=500):
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])

    return chunks

def text2chunk_and_pred(text, summarizer, max_length, min_length):
    sentences = text_preprocessing(text)
    chunks = make_chunks(sentences)

    print(f"전체길이: {len(text)}")
    print(f"chunk개수: {len(chunks)}")
    print(f"chunk하나의 길이: {len(chunks[0])}")

    summary = ""
    for i in range(len(chunks)):
        res = summarizer(chunks[i], max_length=max_length, min_length=min_length, do_sample=False)
        if type(summarizer) == transformers.pipelines.text2text_generation.SummarizationPipeline:
            summary += res[0]["summary_text"]
        else:
            summary += res

    return summary

def isKorean(text):
    hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')  
    result = hangul.findall(text)
    return len(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()

    text = get_text(args.url)
    sentences = text_preprocessing(text)
    chunks = make_chunks(sentences)

    print(f"text 길이: {len(text)}")