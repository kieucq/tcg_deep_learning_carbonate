#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH -J segment_gen
#SBATCH -p gpu --gpus 2
#SBATCH -A r00043 
module load PrgEnv-gnu
module load python/gpu
cd /N/u/ckieu/Carbonate/model/deep-learning/Segment/
sed -i 's/0h/6h/' tcg_segment_p0.py 
python tcg_segment_p0.py
sed -i 's/6h/12h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/12h/18h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/18h/24h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/24h/30h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/30h/36h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/36h/42h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/42h/48h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/48h/54h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/54h/60h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/60h/66h/' tcg_segment_p0.py
python tcg_segment_p0.py
sed -i 's/66h/72h/' tcg_segment_p0.py
python tcg_segment_p0.py



