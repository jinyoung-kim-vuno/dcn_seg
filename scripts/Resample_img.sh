#!/bin/bash

function exit_on_error {
    exit_code=$?
    if [ $? -ne 0 ]; then
        exit $exit_code
    fi
}

input_file=$1
out_file=$2
res=$3

ResampleImage 3 ${1} ${2} ${3} [size=1,spacing=0] 4