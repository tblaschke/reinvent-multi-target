#!/bin/bash
rm -f sampling_jobs

python=`which python`
for config in sampled/sampling_config.*.json ; do
    echo ${python} ${PWD}/reinvent/input.py ${PWD}/${config} >> sampling_jobs
done
