#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "argument:$#" 1>&2
  echo "requirements: 2(detail is below)" 1>&2
  echo "bash install.sh your_anaconda_env_path(~~/envs/number)" 1>&2
  exit 1
fi

prefix="$1"

echo "create conda environment whose name is number"
echo "installing..."

conda env create --file requirements.yml -p $prefix

conda config --remove channels conda-forge
conda config --add channels conda-forge  
conda install ffmpeg
conda install opencv

echo "finished"
echo "activate number before call script"
