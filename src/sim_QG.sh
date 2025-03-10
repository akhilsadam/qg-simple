#!/bin/sh

runid_start=22  # Start value for runid (trained model num)
runid_end=31  # End value for runid


for (( runid=${runid_start}; runid <= ${runid_end}; runid++ )); do
    printf -v padded_m "%04d" "$runid"
    qsub -N "SimQG${padded_m}" "SimQG" "${runid}"
    sleep 3
done
