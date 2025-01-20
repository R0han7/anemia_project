# Use the official Python image as the base image
FROM python:3.9.12
WORKDIR /anemia_project
COPY requirements.txt /anemia_project/
RUN pip install --upgrade pip
COPY requirements.txt ./

RUN pip install -r requirements.txt

EXPOSE 8000
ENV PORT 8080

CMD streamlit run main.py --server.port 8080
