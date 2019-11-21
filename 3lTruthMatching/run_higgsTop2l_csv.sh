while read a; do
    python3.6 higgsTop2lReco.py $a models/xgb_match_top15.dat higgsTop2lFiles &
done<../list3lFiles.txt
