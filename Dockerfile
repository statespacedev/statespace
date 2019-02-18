FROM python:3.6-slim

COPY requirements.txt /statespace/requirements.txt
COPY setup.py /statespace/setup.py
COPY README.md /statespace/README.md
COPY statespace /statespace/statespace
COPY test_statespace.py /statespace/test_statespace.py
COPY pytest.ini /statespace/pytest.ini
ENV PYTHONPATH="/statespace/statespace:$PYTHONPATH"

WORKDIR /statespace

RUN pip install -e .

CMD pytest

#CMD python -m statespace -d

