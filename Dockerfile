# Use the official Python image as the base image
FROM python:3.9.12
EXPOSE 8080
WORKDIR /anemia_project

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=$PORT", "--server.address=0.0.0.0"]

