#!/bin/bash
#SBATCH -o myjob.%j.%N.out
#SBATCH -A hpc1406182255
#SBATCH --partition=C144M4096G
#SBATCH --qos=low
#SBATCH -J RLpsy_Gated_MF
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=124
#SBATCH --mail-type=end
#SBATCH --mail-user=sunshinecltzac@gmail.com
#SBATCH --time=24:00:00

module load anaconda/3-4.4.0.1
source activate lab
export PYTHONPATH=/gpfs/share/home/1400013706/RLpsy:${PYTHONPATH}
cd ~/RLpsy/data_analysis
python Gated_MF_fit.py