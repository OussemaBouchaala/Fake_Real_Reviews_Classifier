FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader -d /usr/local/share/nltk_data stopwords
COPY app ./app
COPY models ./models
COPY config.json .
WORKDIR /srv/app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
