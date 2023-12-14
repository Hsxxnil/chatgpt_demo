PROJECT = $(shell pwd)

setup:
	cp $(PROJECT)/config/config.example $(PROJECT)/config/config_1.py
	pip install -r requirements.txt

train:
	python train_model.py

test:
	python test_model.py