while read a; do
    b=${a#/data_ceph/afwebb/higgs_diff/newRootFiles/3lFiles/}
    b=${b%.*}.csv
    python3.6 findBestHiggs2l.py $a models/xgb_match_higgs2lLepCut.dat models/xgb_match_top3lLepCut.dat outputDataHiggs2l/$b &
done<../list3lFiles.txt
