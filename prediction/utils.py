from bs4 import BeautifulSoup
import requests 
from youtube_transcript_api import YouTubeTranscriptApi
import transformers
import re

def video_extractor(url):
    """
    Args: 
        - url(String)
    Return: 
        - text(String)

    유튜브 포맷의 url 주소를 받아서 자막을 추출하는 함수입니다.
    자막은 한국어를 우선으로 하고 한국어가 없으면 영어를 추출합니다.
    """
    video_id = url.split("=")[1] # 고유의 video id를 url에서 찾습니다.
    try:
        # youtube_transcript_api를 활용하였습니다.
        # 고유의 video_id를 넣고 아래의 dictionary의 나열한 list형태로 자막을 반환합니다.
        # caption = [{"duration": ~, "start": ~, "text": ~}, ...]
        # https://github.com/jdepoix/youtube-transcript-api
        caption = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
    except:
        # 자막이 존재하지 않을 경우 빈 string을 반환합니다.
        return ""

    text = ""
    for i in range(0, len(caption), 1):
        if i % 2 == 1:
            text += '. ' + caption[i]['text']
        else:
            text += ' ' + caption[i]['text']

    return text

def post_extractor(url):
    """
    Args: 
        - post url(String)
    Output: 
        - post text(String)
    """

    #todo:  medium, github(제목, 마크다운 수식), velog, tistory(제목), brunch(공백) 이슈들 수정하기
    r = requests.get(url) # http로 url에 해당하는 request를 받습니다.
    soup = BeautifulSoup(r.text, 'html.parser') # BeautifulSoup으로 html정보를 파싱합니다.

    results = soup.find_all(['p']) # p 태그가 붙어있는 부분을 list형태로 results에 저장합니다.

    # results = soup.find_all(['h1', 'p])
    text_list = [result.text for result in results] # text부분 만을 list로 만듭니다.
    text = ' '.join(text_list)

    return text

def get_text(url):
    """
    Args: 
        - url(String)
    Ouptut: 
        - youtube cation or post text(String)
    """
    if url[:24] == "https://www.youtube.com/":
        text = video_extractor(url)
    elif ("https://medium.com/" in url) or ("tistory.com" in url) or ("https://brunch.co.kr/" in url):
        # todo: 다양한 블로그 사이트 추가하기
        text = post_extractor(url)
    else:
        text = ""
    
    return text

def text_preprocessing(text):
    """
    Args: 
        - raw text(String)
    Output: 
        - preprocessed sentences(List)
    """
    # todo: 데이터 전처리 케이스별 정리
    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>')
    text = text.replace("\xa0", "")
    text = text.replace("\n", "")
    sentences = text.split('<eos>')

    return sentences

def make_chunks(sentences, max_chunk=500):
    """
    Args: 
        - sentences(String)
        - max_chunk(Int)
    Output: 
        - chunks(List)

    메모리 제한을 피하기 위해서 chunk로 분할
    """
    current_chunk = 0 
    chunks = []
    for sentence in sentences: # 개별 문장을 하나씩 진행

        if len(chunks) == current_chunk + 1:
            # 두번째 반복부터

            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                # 새로운 길이를 추가해도 max_chunk보다 작을 경우에 단어 단위로 추가
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                # sentence를 단어단위로 분할해서 하나의 리스트로 만들고 chunks에 추가
                chunks.append(sentence.split(' '))
                # 다음 카운트
                current_chunk += 1
        else:
            # 첫번째 반복일때
            # 문장을 단어단위로 분할해서 새로운 chunk에 추가
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id]) # 공백을 기준으로 모두 합치기

    return chunks

def text2chunk_and_pred(text, summarizer, max_length, min_length):
    """
    Args: 
        - text(String)
        - summarizer(Summarizer)
        - max_length(Int)
        - min_length(Int)
    Output:
        - summary(String)
    """
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
    """
    Args:
        - text(String)
    Output:
        - len(results) (int): 한글의 개수
    """
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