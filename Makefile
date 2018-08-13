.PHONY: venv clean install

activateenv=. venv/bin/activate

venv:
	python3 -m venv venv/

clean:
	rm -rf venv

install:
	$(call activateenv) && pip install -r requirements.txt

reset: clean venv install