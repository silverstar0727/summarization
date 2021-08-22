import chrome_bookmarks
from predict import summarizer

# chrome_bookmarks.folder의 구조는 [1번 폴더 dict, 2번 폴더 dict...]
def add_prediction(children):
    for i in range(len(children)):
        print(children[i])
        # type이 url일때는 summarization하여 "summary"를 key로 하는 예측 추가
        if children[i]["type"] == "url":
            children[i]["summary"] = summarizer(children[i]["url"])
            print(children[i]["summary"])
        # type이 folder일때는 재귀로 하위 항목을 모두 에측하고 그 결과를 children의 i번째에 저장
        else:
            children[i] = add_prediction(children[i]["children"])

    return children

def get_result():
    result = chrome_bookmarks.folders
    for i in range(len(result)):
        result[i]["children"] = add_prediction(result[i]["children"])

    return result

print(get_result())