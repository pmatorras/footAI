help:
	@echo "Available targets:"
	@echo "  download-SP    - Download Spanish data"
	@echo "  elo-SP        - Get ELO for Spain"

# If you run 'make train VERBOSE=yes' it will be 'yes'.
VERBOSE ?= no

# Define a variable that holds the flag if VERBOSE is 'yes', otherwise empty
PYTHON_FLAGS = $(if $(filter $(VERBOSE),yes true 1),-v,)

download-SP:
	footai download --country SP --div SP1
	footai download --country SP --div SP2

elo-SP:
	footai elo --country SP --div SP1
	footai elo --country SP --div SP2

COUNTRY ?= SP
DIVISION ?= SP1,SP2
SEASON_START ?= 23,24
FEATURES_SET ?= baseline
download:
	footai download --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START)
promotion:
	footai promotion-relegation --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -m 

plot_multi:
	footai plot --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m
elo_multi: 
	footai elo --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) -m

test_multi: elo_multi plot_multi

elo:
	footai elo --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  

features_multi:
	footai features --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -v -m

features:
	footai features --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -v

prepare_train: download promotion elo features

train: 
	footai train --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  --features-set $(FEATURES_SET) $(PYTHON_FLAGS)

train_multi: 
	footai train --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  --features-set $(FEATURES_SET) -m $(PYTHON_FLAGS)


train_options: 
	footai train --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  --features-set draw_optimized -m $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  --features-set extended -m $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  --features-set baseline -m $(PYTHON_FLAGS)


plot:
	footai plot --country $(COUNTRY) --div $(DIVISION) --season-start $(SEASON_START) --elo-transfer  -m 

test_elo: elo plot
