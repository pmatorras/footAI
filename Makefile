.PHONY: help
help:
	@echo "FootAI Makefile"
	@echo "==============="
	@echo ""
	@echo "Data Pipeline:"
	@echo "  download              Download match data for countries"
	@echo "  promotion             Process promotion/relegation"
	@echo "  elo                   Calculate ELO ratings"
	@echo "  features              Generate feature sets"
	@echo "  plot                  Visualize ELO ratings"
	@echo ""
	@echo "Dashboard:"
	@echo "  dashboard             Run development server (port 8050)"
	@echo "  dashboard_prod        Run production server with gunicorn (port 8000)"
	@echo "Training:"
	@echo "  train                 Train model (all divisions)"
	@echo "  train_t1              Train model (tier 1 only)"
	@echo "  train_t2              Train model (tier 2 only)"
	@echo "  train_feat_options    Train with all feature sets"
	@echo "  train_models          Train multiple models (requires MODELS=...)"
	@echo "  train_tune            Hyperparameter tuning (production config)"
	@echo ""
	@echo "Pipelines:"
	@echo "  pipeline_plot         download → prepare_elo"
	@echo "  pipeline_train        download → prepare_train → train"
	@echo "  prepare_train         promotion → elo → features"
	@echo ""
	@echo "Configuration Variables:"
	@echo "  COUNTRY=$(COUNTRY)"
	@echo "  SEASON_START=$(SEASON_START)"
	@echo "  MODEL=$(MODEL)"
	@echo "  FEATURES_SET=$(FEATURES_SET)"
	@echo "  VERBOSE=$(VERBOSE)"
	@echo "  MULTI_DIVISION=$(MULTI_DIVISION)"
	@echo ""
	@echo "Examples:"
	@echo "  make train MODEL=lightgbm VERBOSE=yes"
	@echo "  make train_t1 COUNTRY=SP SEASON_START=20-25"
	@echo "  make train_models MODELS='rf lgbm xgb'"

.PHONY: help download promotion elo features train train_t1 train_t2 plot pipeline_plot pipeline_train prepare_train train_models train_tune
#==============================================================================
# CONFIGURATION VARIABLES
#==============================================================================
VERBOSE ?= no
MULTI_DIVISION ?= no
COUNTRY ?= SP,IT,EN,DE,FR
SEASON_START ?= 15-25
FEATURES_SET ?= odds_optimized
MODEL ?= rf
TIERS ?= tier1 tier2
MODELS ?=                      # Space-separated list for train_models (must specify)
#derived variables
DIV_FLAG = $(if $(DIVISION),--div $(DIVISION),)
PYTHON_FLAGS = $(if $(filter $(VERBOSE),yes true 1),-v,)
MULTI_DIVISION := $(if $(MULTI_DIVISION_FLAG),$(MULTI_DIVISION_FLAG),$(MULTI_DIVISION))
MULTI_DIV_FLAG = $(if $(filter $(MULTI_DIVISION),yes true 1),--multi-division,)
FEATURES_SET := $(if $(FEATURES),$(FEATURES),$(FEATURES_SET))

#==============================================================================
# DATA RETRIEVAL & PREPARATION
#==============================================================================

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


#==============================================================================
#  PLOTTING
#==============================================================================
prepare_elo: promotion elo
pipeline_plot: download prepare_elo

plot:
	footai plot --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer -ms 

plot_multi:
	footai plot --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) -ms $(PYTHON_FLAGS)

#==============================================================================
# DASHBOARD
#==============================================================================

dashboard:
	@echo "Starting development dashboard on http://localhost:8050"
	python src/footai/viz/dashboard.py

dashboard_prod:
	@echo "Starting production dashboard on http://localhost:8000"
	gunicorn -b 0.0.0.0:8000 src.footai.viz.dashboard:server


#==============================================================================
# TRAINING TARGETS
#==============================================================================

prepare_train: promotion elo features
prepare_train_multi: promotion elo_multi features_multi

TRAIN_BASE = footai train --country $(COUNTRY) $(DIV_FLAG) --multi-countries --season-start $(SEASON_START) --elo-transfer --model $(MODEL) --features-set $(FEATURES_SET) $(MULTI_DIV_FLAG) $(PYTHON_FLAGS)

train: 
	$(TRAIN_BASE)
train_t1:
	$(TRAIN_BASE) --division=tier1

train_t2:
	$(TRAIN_BASE) --division=tier2


FEATURE_SETS = odds_optimized extended baseline

train_feat_options:
	@$(foreach fs,$(FEATURE_SETS), \
	echo "Training elo transfer using feature set: $(fs)"; \
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) --elo-transfer $(MULTI_DIV_FLAG) --features-set $(fs) --model $(MODEL) -ms $(PYTHON_FLAGS);)

train_feat_multi_options:
	@$(foreach fs,$(FEATURE_SETS), \
	echo "Training multi options using  feature set: $(fs)"; \
	footai train --country $(COUNTRY) $(DIV_FLAG) --season-start $(SEASON_START) $(MULTI_DIV_FLAG) --features-set $(fs) --model $(MODEL) -ms $(PYTHON_FLAGS);)


pipeline_train: download promotion elo features train
train_models:
	@$(foreach MODEL,$(MODELS), \
		echo "\n########################################\n"; \
		echo "\n#### Training model: $(MODEL) ####\n"; \
		echo "\n########################################\n"; \
		footai train --country SP,IT,EN,DE,FR --season-start 15-25 --elo-transfer --multi-countries --division=tier1 --model $(MODEL) --multi-division --features-set $(FEATURES_SET);)


train_tune:
	echo "Tuning all tiers combined..."
	footai train --country SP,IT,EN,DE,FR --season-start 15-25 --elo-transfer --multi-countries --model rf --multi-division --features-set odds_optimized --tune --tune-iterations=100
	@$(foreach tier,$(TIERS), \
	echo "Training with feature set: $(tier)"; \
	footai train --country SP,IT,EN,DE,FR --season-start 15-25 --elo-transfer --multi-countries --model rf --multi-division --features-set odds_optimized --division $(tier) --tune --tune-iterations=100 ;)
