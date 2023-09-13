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
python extract_environment_features_from_ncep_13vars_grib2.py /N/project/hurricane-deep-learning/data/ncep-fnl/ /N/project/hurricane-deep-learning/data/ncep_extracted_41x161_13vars_br200 --lat 5 45 --lon 100 260
