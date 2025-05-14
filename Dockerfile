FROM python:3.10.16-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libopencv-dev && apt-get clean
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "script.py"]
