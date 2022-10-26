# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN sed -i 's#http://deb.debian.org#https://mirrors.163.com#g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN echo &&git clone https://github.com/bforecast/strategy-app.git .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --upgrade pip

# TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]