#!/bin/bash

function exit_on_error {
    local exit_code=$?
    if [ $? -ne 0 ]; then
        exit "${exit_code}"
    fi
}

ref_file=$1
input_file=$2
out_file=$3
affine_iter_num=$4
mask=$5

if [ -z "${mask}" ]; then
    mask_options=""
else
    mask_options="-x ${mask}"
fi

ANTS 3 -m MI[${ref_file},${input_file},1,64] -o ${out_file}_ --number-of-affine-iterations ${affine_iter_num} -i 0 ${mask_options} -a I.txt
exit_on_error