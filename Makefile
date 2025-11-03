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