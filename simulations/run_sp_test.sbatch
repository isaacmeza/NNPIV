#!/bin/bash
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH -p test
#SBATCH -t 0-11:30  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 110        # Number of cores (-c)
#SBATCH --mem=300000
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mail-user=i.mezal.12@gmail.com #Email to which notifications will be sent

# load modules
module load python/3.10.13-fasrc01
mamba activate nnpiv_env

NODE_ID=0
N_NODES=1
CONFIG=$1
tmpdir='temp'
tmpfile="${NODE_ID}_${CONFIG}"
mkdir -p "${tmpdir}"
cp ${CONFIG}.py ${tmpdir}/${tmpfile}.py
sed -i s/__NODEID__/${NODE_ID}/g ${tmpdir}/${tmpfile}.py
sed -i s/__NNODES__/${N_NODES}/g ${tmpdir}/${tmpfile}.py
srun -c 110 python sweep_sp.py --config ${tmpdir}.${tmpfile}
rm ${tmpdir}/${tmpfile}.py