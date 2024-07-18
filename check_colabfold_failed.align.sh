#!/bin/bash

base_path=$1

echo "running on base path: $base_path"

list=$(ls -d ${base_path}/*/)

echo "Here is the list of failed align jobs:"
for nn in $list
do
  n=$(basename ${nn})
  if [ ! -f ${n}/0.a3m ]; then
    g=$(grep -e ${n} 01_submit_all_colab_search_jobs.sh)
    echo "${g}"
  fi
done

# detect failed fold jobs
list=$(ls -d ${base_path}/*/)
echo ""
echo ""
echo "Here is the list of failed fold jobs:"
for nn in $list
do
  n=$(basename ${nn})
  if [ ! -f ${n}.zip ]; then
#    echo "$n"
    g=$(grep -e ${n} 02_submit_all_colab_fold_jobs.sh)
    echo "${g}"
  fi
done


