FROM python:3.10-slim

WORKDIR /app

# Cài các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Sao chép mã nguồn
COPY . .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
