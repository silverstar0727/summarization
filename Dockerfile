FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD [ "python" "summarization.py" "--url" "https://medium.com/syncedreview/google-t5-explores-the-limits-of-transfer-learning-a87afbf2615b"]

