help:
	@echo "Available targets:"
	@echo "  download-SP    - Download Spanish data"
	@echo "  elo-SP        - Get ELO for Spain"



download-SP:
	python main.py download --country SP --div SP1
	python main.py download --country SP --div SP2

elo-SP:
	python main.py elo --country SP --div SP1
	python main.py elo --country SP --div SP2

COUNTRY ?= SP
DIVISION ?= SP1
SEASON_START ?= 22,23,24,25

test_elo:
	python main.py download --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m
	python main.py elo --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m
	python main.py plot --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m
