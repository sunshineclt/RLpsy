#!/bin/bash
#SBATCH -o MF_no_reinit.%j.%N.out
#SBATCH -A hpc1406182255
#SBATCH --partition=C032M0128G
#SBATCH --qos=low
#SBATCH -J RLpsy_MF_no_reinit
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=end
#SBATCH --mail-user=sunshinecltzac@gmail.com
#SBATCH --time=24:00:00

module load anaconda/3-4.4.0.1
source activate lab
export PYTHONPATH=/gpfs/share/home/1400013706/RLpsy:${PYTHONPATH}
cd ~/RLpsy/data_analysis
python MF_no_reinit_fit.py