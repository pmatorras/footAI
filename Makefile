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
DIVISION ?= SP1,SP2
SEASON_START ?= 23,24
download:
	python main.py download --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START)
promotion:
	python main.py promotion-relegation --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -m 

plot_multi:
	python main.py plot --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m
elo_multi: 
	python main.py elo --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m

test_multi: elo_multi plot_multi

elo:
	python main.py elo --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -m 
plot:
	python main.py plot --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -m 

test_elo: elo plot
