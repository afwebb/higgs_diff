while read a; do
    b=${a#/data_ceph/afwebb/higgs_diff/newRootFiles/3lFilesJVTCut/}
    b=${b%.*}.csv
    #python3.6 findBestHiggs2l.py $a models/xgb_match_higgs2l.dat models/xgb_match_top3l.dat outputDataHiggs2l/$b &
    python3.6 findBestHiggs2lKeras.py $a models/keras_model_higgs2l.h5 models/keras_model_top3l.h5 outputDataHiggs2l/$b &
done<../list3lFiles.txt
