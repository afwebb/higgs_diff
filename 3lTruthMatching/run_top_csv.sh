while read a; do
    python3.6 topReco.py $a &
done<../list3lFiles.txt
