FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY music_genre_cnn.h5 .
COPY scaler.joblib .

RUN pip install -r requirements.txt
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]