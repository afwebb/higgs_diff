for nj in 4 6
do
    for mc in a d
    do
	for dsid in 345672 345673 345674
	do
	    echo python loop_events.py /data_ceph/afwebb/datasets/v06_21/GN1/mc16$mc/Nominal/$dsid.root mc16${mc}_${dsid}_${nj}j.csv $nj &
	done 
    done
done

