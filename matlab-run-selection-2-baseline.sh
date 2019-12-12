#!/bin/bash
set -o errexit
module load matlab
cd gaussian_process/matlab

# The function run_sample_selection_2() is called with the following arguments
# 1. Path to the test fold
# 2. The test fold number - integer: this is typically the same as the last part of the above path, but given as an argument so that they can be different.
# 3. The maximum number of samples to be selected. For testing, 10 is good.  For final run: perhaps 60 or 80; the larger it is, the much longer it takes to run.
# 4. The postfix to the result folder (used to indicate the sampling method)
# 5. The sampling metric function (see file gaussian_process/matlab/sampling_metrics.org or options below)
# 6. A boolean value whether Euclidean distance metric is used (true if used, false if not; default is false).

# Method: maxvar-train
MYPOSTFIX='maxvar-train'
MYSAMPLING='@samplingmetric_maxvartrain'
MYEUCL=false
# For other methods, select the options below

# Method: var*-dist*-full
MY_N=0
MY_NSTR="full"

# Method: var*-dist*-maxvar
MY_N=1
MY_NSTR="maxvar"

# Method: var*-dist*-top<N> (replace <N> with actual number)
MY_N=2
MY_NSTR="top$MY_N"

# Select 'min' or 'ave' for distance type
MYDISTTYPE="min"   # or "ave"

# Select alpha for variance, and set the prefix accordingly (prefix is empty if alpha=0)
MY_ALPHA=1
MY_ALPHASTR="var${MY_ALPHA}-"  # MY_ALPHASTR="" if MY_ALPHA=0

# Select distance metric option: Euclidean or not
MYEUCL=false     # or true

if [ "$MYEUCL" == 'true' ]
then
	MYEUCLSTR='eucl'
else
	MYEUCLSTR=''
fi

MYPOSTFIX="${MY_ALPHASTR}${MYEUCLSTR}dist${MYDISTTYPE}-${MY_NSTR}"
MYSAMPLING="@(x,y,z) samplingmetric_vardist(x,y,z,${MY_ALPHA},'${MYDISTTYPE}',${MY_N})"

echo $MYPOSTFIX
echo $MYSAMPLING
echo "run gpml-matlab-v4.2-2018-06-11/startup.m; run_baseline_sample_selection_2('/projects/genomic-ml/neuroblastoma-data/data/H3K36me3_TDH_ENCODE/cv/equal_labels/testFolds/1', 1, 60, '$MYPOSTFIX', $MYSAMPLING, $MYEUCL);exit;"

matlab -nodisplay -nosplash -nodesktop -r "run gpml-matlab-v4.2-2018-06-11/startup.m; run_baseline_sample_selection_2('/projects/genomic-ml/neuroblastoma-data/data/H3K36me3_TDH_ENCODE/cv/equal_labels/testFolds/1', 1, 60, '$MYPOSTFIX', $MYSAMPLING, $MYEUCL);exit;"
