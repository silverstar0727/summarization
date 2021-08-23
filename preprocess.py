from bs4 import BeautifulSoup
import requests 
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

# input: youtube url -> output: video title, transcript text
def video_extractor(url):
    yt = YouTube(url)
    title = yt.title

    video_id = url.split("=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])

    text = ""
    for sub_script in transcript:
        text += '. ' + sub_script['text']

    text = text.replace("\xa0", "")
    text = text.replace("\n", "")

    return title, text

# input: post url -> output: post title, post text
def post_extractor(url):
    #todo:  medium, github(제목 이슈, 마크다운 수식 이슈), velog, tistory(제목이슈), brunch(공백이슈)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    title = soup.find_all('h1')[0].text
    results = soup.find_all(['p'])

    # results = soup.find_all(['h1', 'p])
    text = [result.text for result in results]
    text = ' '.join(text)

    return title, text

# get title and text from url
def get_title_and_text(url):
    if url[:24] == "https://www.youtube.com/":
        title, text = video_extractor(url)
    elif ("https://medium.com/" in url) or ("tistory.com" in url) or ("https://brunch.co.kr/" in url):
        title, text = post_extractor(url)
    else:
        title = ""
        text = ""
    
    return title, text

def text_preprocessing(text):
    # todo: 데이터 전처리 케이스별 정리
    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>')
    sentences = text.split('<eos>')

    return sentences

# 메모리 제한을 피하기 위해서 chunk로 분할
def make_chunks(sentences, max_chunk=4096):
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="blog url")
    args = parser.parse_args()

    title, text = get_title_and_text(args.url)
    sentences = text_preprocessing(text)
    chunks = make_chunks(sentences)

    print(f"text 길이: {len(text)}")