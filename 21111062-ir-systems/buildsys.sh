#!/bin/bash
jupyter-nbconvert building_system.ipynb --to python
chmod +x building_system.py
python3 building_system.py
