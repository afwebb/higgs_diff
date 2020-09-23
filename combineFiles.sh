a=$1
head -n 1 $a/mc16a/345875.csv > $a/totalRank.csv 
tail -q -n +2 $a/mc16*/346*.csv >> $a/totalRank.csv
#tail -q -n +2 $a/4*.csv >> $a/total.csv
