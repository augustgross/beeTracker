VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

ifeq ($(OS),Windows_NT)
	PYTHON = $(VENV)/Scripts/python
	PIP = $(VENV)/Scripts/pip
endif

.PHONY: setup run clean

setup: $(VENV)/installed

$(VENV)/installed:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install ultralytics opencv-python pandas scipy matplotlib openpyxl
	touch $(VENV)/installed

run: setup
	$(PYTHON) tracker.py

clean:
	rm -rf $(VENV)
	rm -f bee_telemetry.csv bee_telemetry_with_speed.csv
	rm -f bee_summary_statistics.xlsx speed_plot.png tracked_video.mp4