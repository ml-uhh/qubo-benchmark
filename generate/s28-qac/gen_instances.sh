#!/bin/bash


seedstxt=`cat seeds.txt`

llist=`seq 3 15`
llist=($llist)
randseeds=($seedstxt)
# number of instances
ninst=125

for i in "${!llist[@]}"; do
	l=${llist[$i]}
	seed=${randseeds[$i]}
	outdir=$l
	mkdir -p $outdir
	echo Generating l=$l ...
	for n in `seq 0 $ninst`; do
		bash ./gen_single.sh s28 $l $n  $outdir $seed
	done
done	
