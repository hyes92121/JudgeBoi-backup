FROM pytorch/pytorch

WORKDIR /torch-server 
COPY requirements.txt /torch-server/

RUN pip install -r requirements.txt 
RUN apt update
RUN apt install --assume-yes libgtk2.0-dev
