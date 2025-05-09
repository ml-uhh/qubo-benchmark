#!/bin/bash

topdir=../p-qac-top

# instance class
inst=$1
# instance size
l=$2
# instance index
n=$3
# output directory
outdir=$4
# random seed
sd=$5


flnm=$outdir/qac_L${l}_${n}.txt
if [ -f $flnm ]; then
	echo "Skipping $flnm";
	exit
fi

echo Writing instance $n to $flnm
pgt-gen -n $n  \
	--seed $sd \
	$topdir/pqac_L${l}.txt $inst $flnm

