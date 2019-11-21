while read a; do
    python3.6 higgs2lReco.py $a higgs2lFiles &
done<../list3lFiles.txt
