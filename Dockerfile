# Use the official Python image as the base image
FROM python:3.9.12
EXPOSE 8080
EXPOSE 8501
WORKDIR /anemia_project
ENV PORT 8501

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=$PORT", "--server.address=0.0.0.0"]

