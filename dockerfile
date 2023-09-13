FROM python:3.8
ENV TZ=Asia/Jakarta

WORKDIR /opt/app

COPY requirements.txt ./
RUN pip install -r requirements.txt


COPY . ./
EXPOSE 8072
CMD ["python", "main.py"]