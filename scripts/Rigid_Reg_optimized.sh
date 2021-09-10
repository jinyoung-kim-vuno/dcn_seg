#!/bin/bash

function exit_on_error {
    exit_code=$?
    if [ $? -ne 0 ]; then
        exit $exit_code
    fi
}

ref_file=$1
input_file=$2
out_file=$3
affine_iter_num=$4

ANTS 3 -m MI[$ref_file,$input_file,1,32] -o ${out_file}_ --number-of-affine-iterations ${affine_iter_num=$4} -i 0 --rigid-affine true
exit_on_error
