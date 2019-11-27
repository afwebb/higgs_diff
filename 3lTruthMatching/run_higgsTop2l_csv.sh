while read a; do
    python3.6 higgsTop2lReco.py $a models/xgb_match_top14.dat higgsTop2lFiles &
done<../list3lFiles.txt
