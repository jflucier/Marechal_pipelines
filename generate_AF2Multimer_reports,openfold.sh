#!/bin/bash

base_path=$1
base_path_name=$(basename ${base_path})

echo "running on base path: $base_path"

echo "generating ${base_path_name}.interfaces.csv..."
for f in ${base_path}/*/predictions/*_analysis/interfaces.csv
do
  t=$(dirname $f)
  tt=$(dirname $t)
  name=$(basename $tt)
#  echo "$f --> $name"
  perl -ne '
  chomp($_);
  if($_ !~ /^complex_name/){
    print "'${name}'," . $_ . "\n";
  }
  ' $f > ${f}.foldname.csv
done

echo "fold_name,complex_name,model_num,pdockq,ncontacts,plddt_min,plddt_avg,plddt_max,pae_min,pae_avg,pae_max,distance_avg" > ${base_path}/${base_path_name}.interfaces.csv
cat ${base_path}/*/predictions/*_analysis/interfaces.csv.foldname.csv >> ${base_path}/${base_path_name}.interfaces.csv

echo "generating ${base_path_name}.summary.csv..."
for f in ${base_path}/*/predictions/*_analysis/summary.csv
do
  t=$(dirname $f)
  tt=$(dirname $t)
  name=$(basename $tt)
#  echo "$f --> $name"
  perl -ne '
  chomp($_);
  if($_ !~ /^complex_name/){
    print "'${name}'," . $_ . "\n";
  }
  ' $f > ${f}.foldname.csv
done

echo "fold_name,complex_name,avg_n_models,max_n_models,num_contacts_with_max_n_models,num_unique_contacts,best_model_num,best_pdockq,best_plddt_avg,best_pae_avg" > ${base_path}/${base_path_name}.summary.csv
cat ${base_path}/*/predictions/*_analysis/summary.csv.foldname.csv >> ${base_path}/${base_path_name}.summary.csv

echo "generating ${base_path_name}.contacts.csv..."
for f in ${base_path}/*/predictions/*_analysis/contacts.csv
do
  t=$(dirname $f)
  tt=$(dirname $t)
  name=$(basename $tt)
#  echo "$f --> $name"
  perl -ne '
  chomp($_);
  if($_ !~ /^complex_name/){
    print "'${name}'," . $_ . "\n";
  }
  ' $f > ${f}.foldname.csv
done

echo "fold_name,complex_name,model_num,aa1_chain,aa1_index,aa1_type,aa1_plddt,aa2_chain,aa2_index,aa2_type,aa2_plddt,pae,min_distance" > ${base_path}/${base_path_name}.contacts.csv
cat ${base_path}/*/predictions/*_analysis/contacts.csv.foldname.csv >> ${base_path}/${base_path_name}.contacts.csv

echo "done!"