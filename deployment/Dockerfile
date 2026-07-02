FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Run both (in production you might separate)
CMD ["streamlit", "run", "frontend/streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]