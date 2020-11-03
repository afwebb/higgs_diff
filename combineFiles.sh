a=$1
head -n 1 $a/mc16a/345875.csv > $a/total.csv 
tail -q -n +2 $a/mc16*/*.csv >> $a/total.csv
#tail -q -n +2 $a/4*.csv >> $a/total.csv
