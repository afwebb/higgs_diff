a=$1
head -n 1 $a/345874dFlat.csv > $a/totalFlat.csv 
tail -q -n +2 $a/34*Flat.csv >> $a/totalFlat.csv
