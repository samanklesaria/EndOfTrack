#!/bin/sh
nvidia-smi | gawk -F '|' 'match($4, /([0-9]+)%/, a) {print a[1]}'
