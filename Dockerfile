FROM python:3.8

EXPOSE 5432

COPY requirements.txt .

RUN apt update
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app

CMD python main.py
