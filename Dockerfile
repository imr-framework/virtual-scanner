FROM continuumio/miniconda3
# docker build -t virtualscanner .
RUN apt-get update && \
    apt-get install -y libglib2.0-0 \
                       libsm6 libxext6 libxrender-dev && \
    mkdir -p /code
ADD . /code
RUN pip install -r /code/requirements.txt
WORKDIR /code
RUN python setup.py install
EXPOSE 5000
ENTRYPOINT ["/opt/conda/bin/virtualscanner"]
