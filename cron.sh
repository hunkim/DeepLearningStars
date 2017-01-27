#!/bin/sh
git pull
python3 list2md.py
git commit -m"Auto update" -a
git push origin
