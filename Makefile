help:
	@echo "Available targets:"
	@echo "  download-SP    - Download Spanish data"
	@echo "  elo-SP        - Get ELO for Spain"

# If you run 'make train VERBOSE=yes' it will be 'yes'.
VERBOSE ?= no
MULTI_DIVISION ?= no
MULTI_DIVISION := $(if $(MULTI_DIVISION_FLAG),$(MULTI_DIVISION_FLAG),$(MULTI_DIVISION))
# Define a variable that holds the flag if VERBOSE is 'yes', otherwise empty
PYTHON_FLAGS = $(if $(filter $(VERBOSE),yes true 1),-v,)
MULTI_DIV_FLAG = $(if $(filter $(MULTI_DIVISION),yes true 1),--multi-division,)
COUNTRY ?= SP
SEASON_START ?= 22,23,24
FEATURES_SET ?= baseline
FEATURES_SET := $(if $(FEATURES),$(FEATURES),$(FEATURES_SET))

MODEL ?= rf

# Optional: Allow overriding if needed
DIV_FLAG = $(if $(DIVISION),--div $(DIVISION),)

download:
	footai download --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START)

promotion:
	footai promotion-relegation --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer -ms $(PYTHON_FLAGS)

elo:
	footai elo --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(PYTHON_FLAGS)

elo_multi: 
	footai elo --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) -ms $(PYTHON_FLAGS)
elo_nomulti: 
	footai elo --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(PYTHON_FLAGS)
features_multi:
	footai features --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(PYTHON_FLAGS) -ms

features_nomulti:
	footai features --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(PYTHON_FLAGS)

features:
	footai features --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(PYTHON_FLAGS)

prepare_train: promotion elo features
prepare_train_multi: promotion elo_multi features_multi

train: 
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer --model $(MODEL)  --features-set  $(FEATURES_SET)  $(MULTI_DIV_FLAG) $(PYTHON_FLAGS)

train_multi: 
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --model $(MODEL) --features-set $(FEATURES_SET) -ms $(PYTHON_FLAGS)

train_options: 
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(MULTI_DIV_FLAG) --features-set draw_optimized --model $(MODEL) -ms $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(MULTI_DIV_FLAG) --features-set extended --model $(MODEL) -ms $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(MULTI_DIV_FLAG) --features-set baseline --model $(MODEL) -ms $(PYTHON_FLAGS)

train_multi_options: 
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(MULTI_DIV_FLAG) --features-set draw_optimized --model $(MODEL) -ms $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(MULTI_DIV_FLAG) --features-set extended --model $(MODEL) -ms $(PYTHON_FLAGS)
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(MULTI_DIV_FLAG) --features-set baseline --model $(MODEL) -ms $(PYTHON_FLAGS)

plot:
	footai plot --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer -ms 

plot_multi:
	footai plot --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) -ms $(PYTHON_FLAGS)

test_multi: elo_multi plot_multi
test_elo: elo plot

pipeline: download promotion elo features train
test_ranges: elo features train
