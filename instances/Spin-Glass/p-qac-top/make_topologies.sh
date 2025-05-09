#!/bin/bash

llist=( 2 3 4 5 6 7 8 9 10 11 12 13 14 15 )

for l in ${llist[@]}; do
  echo $l
  nm=./pqac_L${l}
  pgt-qac-top -L $l --labels ${nm}_labels.json --plot ${nm}_plot.png --graphml ${nm}.graphml ${nm}.txt
done
