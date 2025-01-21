# Use the official Python image as the base image
FROM python:3.9.12

EXPOSE 8501

WORKDIR /anemia=project

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run main.py
