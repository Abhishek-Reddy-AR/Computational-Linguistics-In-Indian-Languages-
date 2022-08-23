#!/bin/bash
jupyter-nbconvert running_system.ipynb --to python
chmod +x running_system.py
python3 running_system.py $1
