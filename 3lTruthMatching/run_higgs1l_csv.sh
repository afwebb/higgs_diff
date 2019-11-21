while read a; do
    python3.6 higgs1lReco.py $a higgs1lFiles &
done<../list3lFiles.txt
