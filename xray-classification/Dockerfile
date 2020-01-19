FROM python:3.7.5

WORKDIR /src

COPY requirements.txt /src/
RUN pip install -r requirements.txt

COPY model0.py, model1.py, model2.py, model3.py, model4.py server.py /src/

RUN useradd docker
USER docker

EXPOSE 6324

CMD ["python", "server.py"]
