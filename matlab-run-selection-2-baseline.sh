#!/bin/bash
set -o errexit
module load matlab
cd gaussian_process/matlab

# The function run_sample_selection_2() is called with the following arguments
# 1. Path to the test fold
# 2. The test fold number - integer: this is typically the same as the last part of the above path, but given as an argument so that they can be different.
# 3. The maximum number of samples to be selected. For testing, 10 is good.  For final run: perhaps 60 or 80; the larger it is, the much longer it takes to run.

matlab -nodisplay -nosplash -nodesktop -r "run gpml-matlab-v4.2-2018-06-11/startup.m; run_baseline_sample_selection_2('/projects/genomic-ml/neuroblastoma-data/data/H3K36me3_TDH_ENCODE/cv/equal_labels/testFolds/1', 1, 60);exit;"

