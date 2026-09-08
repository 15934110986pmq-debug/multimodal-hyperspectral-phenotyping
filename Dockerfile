FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 系统依赖：点云/GeoTIFF 相关库所需
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
