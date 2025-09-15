#!/bin/bash
python -m cProfile -o temp.dat thrack/onepool_data_demo.py
snakeviz temp.dat