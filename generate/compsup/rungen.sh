mkdir -p "./instances/generated/biclique_(2, 16, 16)_precision256/"
for i in {0..19}
do
    python generate.py --topology biclique --dimensions 2 16 16 --seed $i > "./instances/generated/biclique_(2, 16, 16)_precision256/seed$i.csv"
done
