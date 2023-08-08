FROM python:3.10.10

WORKDIR /app

RUN mkdir /app/logs

COPY scripts/process.py /app/

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY mlhousing-1.0-py3-none-any.whl /tmp/

RUN pip install /tmp/mlhousing-1.0-py3-none-any.whl

ENV MLFLOW_TRACKING_URI=file:/app/mlflow_data

CMD ["python", "process.py"]

