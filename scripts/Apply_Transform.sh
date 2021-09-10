#!/bin/bash

input_file=$1
ref_file=$2
out_file=$3
list_of_transforms=$4

WarpImageMultiTransform 3 ${input_file} ${out_file} -R ${ref_file} ${list_of_transforms} --use-BSpline