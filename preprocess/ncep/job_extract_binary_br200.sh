#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 36:00:00
#SBATCH -J ncep_extract
#SBATCH -p gpu --gpus 2
#SBATCH -A r00043 
conda activate tc
cd /N/u/ckieu/Carbonate/model/deep-learning-quan/scripts
rm -f /N/project/hurricane-deep-learning/data/ncep-fnl/*/*.idx
rm -rf /N/project/hurricane-deep-learning/data/ncep_extracted_41x161_13vars_br200
python create_ncep_binary.py --best-track /N/project/pfec_climo/qmnguyen/tc_prediction/data/bdeck/ibtracs.ALL.list.v04r00.csv --ncep-fnl /N/slate/ckieu/tmp/input/ --basin WP --leadtime 0 --domain-size 20 --distance 45 --output /N/slate/ckieu/tmp/output/test_20x20
