a=$1
head -n 1 $a/345874a.csv > $a/total.csv 
tail -q -n +2 $a/3*.csv >> $a/total.csv
tail -q -n +2 $a/4*.csv >> $a/total.csv
