from predict import Summarizer
import json

# chrome_bookmarks.folder의 구조는 [1번 폴더 dict, 2번 폴더 dict...]
def add_prediction(summarizer, children):
    for i in range(len(children)):
        # type이 url일때는 summarization하여 "summary"를 key로 하는 예측 추가
        if children[i]["type"] == "url":
            children[i]["summary"] = summarizer.predict(children[i]["url"])
            print(f"{children[i]}: {children[i]['summary']}")

        # type이 folder일때는 재귀로 하위 항목을 모두 에측하고 그 결과를 children의 i번째에 저장
        else:
            children[i] = add_prediction(children[i]["children"])
            print(f"{children[i]}: is folder")

    return children

def get_result(summarizer, bookmarks_path="./local/bookmarks.json"):
    with open(bookmarks_path, 'r') as f:
        result = json.load(f)
    for i in range(len(result)):
        result[i]["children"] = add_prediction(summarizer, result[i]["children"])

    return result

def result_2_json(result, result_path):
    with open(result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    summarizer = Summarizer("t5-large")
    summarizer.get_model()
    result = get_result(summarizer)