#!/usr/bin/env bash

echo "create conda environment whose name is number"
echo "installing..."

conda create --file requirements.yaml -n number

echo "finished"
echo "activate number before call script"
