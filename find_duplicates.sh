#!/usr/bin/bash
FILEPATH=list.txt

sort $FILEPATH | uniq --count --repeated