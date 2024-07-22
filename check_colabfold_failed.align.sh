#!/bin/bash

check_op=$1
base_path=$2

echo "running on base path: $base_path"
list=$(ls -d ${base_path}/*/)

if [[ "$check_op" == "align" ]]; then
  echo "Here is the list of failed align jobs:"
  for nn in $list
  do
    n=$(basename ${nn})
    if [ ! -f ${nn}/0.a3m ]; then
      g=$(grep -e ${n} ${base_path}/01_submit_all_colab_search_jobs.sh)
      echo "${g}"
    fi
  done
elif [[ "$check_op" == "fold" ]]; then
  echo "Here is the list of failed fold jobs:"
  for nn in $list
  do
    n=$(basename ${nn})
    if [ ! -f ${nn}/${n}.zip ]; then
      g=$(grep -e ${n} ${base_path}/02_submit_all_colab_fold_jobs.sh)
      echo "${g}"
    fi
  done
else
   echo "Unrecongnised check operation. Possible values are: align or fold"
fi



