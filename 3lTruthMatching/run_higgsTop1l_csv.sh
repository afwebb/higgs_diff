while read a; do
    python3.6 higgsTop1lReco.py $a models/xgb_match_top14.dat higgsTop1lFiles &
done<../list3lFiles.txt
