#!/usr/bin/env bash

git pull --rebase
shift
python3 game.py $@