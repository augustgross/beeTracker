VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

ifeq ($(OS),Windows_NT)
	PYTHON = $(VENV)/Scripts/python
	PIP = $(VENV)/Scripts/pip
endif

.PHONY: setup run clean

help:
	@echo "Available targets:"
	@echo "  setup - Set up the virtual environment and install dependencies"
	@echo "  run   - Run the bee tracker script"
	@echo "  run-live - Run the live bee tracker using the camera feed"
	@echo "  clean - Remove the virtual environment and generated files"

setup: $(VENV)/installed

$(VENV)/installed:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/installed

run: setup
	$(PYTHON) tracker.py

run-live: 
	$(PYTHON) live_tracker.py --camera 0

clean:
	rm -rf $(VENV)
	rm -f bee_telemetry.csv bee_telemetry_with_speed.csv
	rm -f bee_summary_statistics.xlsx speed_plot.png tracked_video.mp4