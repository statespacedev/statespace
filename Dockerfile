FROM python:3.6-slim
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY README.md /README.md
ADD statespace /statespace
RUN pip install -e .
ENV PYTHONPATH="/statespace:$PYTHONPATH"
CMD python -m statespace -d

