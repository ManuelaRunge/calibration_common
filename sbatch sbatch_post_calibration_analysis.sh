#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name="postcalib"
#SBATCH --error=log/postcalib.%j.err
#SBATCH --output=log/postcalib.%j.out


module purge all
source activate /projects/b1139/environments/emod_torch_tobias
cd "$(dirname "$0")"

python post_calibration_analysis.py --experiment '241013_20_max_infections' --length_scales_by_objective True --plot_length_scales True --plot_predictions True --exclude_count 1000 --plot_timers True
