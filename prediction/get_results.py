from Summarizer import Summarizer
import json

# chrome_bookmarks.folder의 구조는 [1번 폴더 dict, 2번 폴더 dict...]
def add_prediction(summarizer, children):
    for i in range(len(children)):
        # type이 url일때는 summarization하여 "summary"를 key로 하는 예측 추가합니다.
        if children[i]["type"] == "url":
            short_sum, long_sum = summarizer.predict(children[i]["url"])
            children[i]["short_summary"], children[i]["long_summary"] = short_sum, long_sum
            print(children[i]["url"])

            if long_sum == "":
                continue

        # type이 folder일때는 재귀로 하위 항목을 모두 에측하고 그 결과를 children의 i번째에 저장합니다.
        else:
            children[i] = add_prediction(summarizer, children[i]["children"])
            print(f"is folder")

    return children

def get_result(summarizer, bookmarks_path="./summarization/local/bookmarks.json"):
    with open(bookmarks_path, 'r') as f:
        result = json.load(f)
    for i in range(len(result)):
        result[i]["children"] = add_prediction(summarizer, result[i]["children"])

    return result

# 결과를 json 파일로 저장합니다.
def result_2_json(result, result_path="./result.json"):
    with open(result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    summarizer = Summarizer("t5-large")
    result = get_result(summarizer)
    result_2_json(result)