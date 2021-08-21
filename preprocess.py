from bs4 import BeautifulSoup
import requests 

def text_extraction(url):
    #todo:  medium, github(제목 이슈, 마크다운 수식 이슈), velog, tistory(제목이슈), brunch(공백이슈)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    title = soup.find_all('h1')[0].text
    results = soup.find_all(['p'])

    # results = soup.find_all(['h1', 'p])
    text = [result.text for result in results]
    article = ' '.join(text)

    return title, article

def data_preprocessing(article):
    # todo: 데이터 전처리 케이스별 정리
    article = article.replace('.', '.<eos>')
    article = article.replace('?', '?<eos>')
    article = article.replace('!', '!<eos>')
    sentences = article.split('<eos>')

    return sentences

def make_chunks(sentences, max_chunk=512):
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

    title, article = text_extraction(args.url)
    sentences = data_preprocessing(article)
    chunks = make_chunks(sentences)

    print(f"article 길이: {len(article)}")
    print(f"chunk 길이: {len(chunks)}")