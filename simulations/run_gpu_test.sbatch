#!/bin/bash
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH -p gpu_test
#SBATCH -t 0-10:00  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 32        # Number of cores (-c)
#SBATCH --gres=gpu:4 # Number of GPUs
#SBATCH --gpu-freq=high # GPU Frequency
#SBATCH --mem=100000
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mail-user=i.mezal.12@gmail.com #Email to which notifications will be sent



# load modules
module load python/3.10.9-fasrc01
mamba activate python_envgpu


NODE_ID=0
N_NODES=1
CONFIG=$1
tmpdir='temp'
tmpfile="${NODE_ID}_${CONFIG}"
mkdir -p "${tmpdir}"
cp ${CONFIG}.py ${tmpdir}/${tmpfile}.py
sed -i s/__NODEID__/${NODE_ID}/g ${tmpdir}/${tmpfile}.py
sed -i s/__NNODES__/${N_NODES}/g ${tmpdir}/${tmpfile}.py
srun -c 16 --gres=gpu:4  python sweep_mc_from_config.py --config ${tmpdir}.${tmpfile}
rm ${tmpdir}/${tmpfile}.py