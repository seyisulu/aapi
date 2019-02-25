FROM python:3.7.2-alpine3.9

RUN mkdir /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -U -r requirements.txt

ENV CORPORA stopwords
RUN python -m nltk.downloader $CORPORA

ENV PUNKT punkt
RUN python -m nltk.downloader $PUNKT

ENV APT averaged_perceptron_tagger
RUN python -m nltk.downloader $APT

ENV WORDNET wordnet
RUN python -m nltk.downloader $WORDNET

COPY . /app

CMD ["python3", "main.py"]
