#!/usr/bin/env bash
function exit_on_error {
    exit_code=$?
    if [ $? -ne 0 ]; then
        exit $exit_code
    fi
}

ref_file=$1
input_file=$2
out_file=$3

ThresholdImage 3 ${ref_file} ${out_file} Otsu 3

MultiplyImages 3 ${out_file} ${input_file} ${out_file}

ThresholdImage 3 ${out_file} ${out_file} Otsu 3

ThresholdImage 3 ${out_file} ${out_file} 1 3 1 0

ImageMath 3 ${out_file} MD ${out_file} 10 # Morphological Delineation with radius 20 (close holes)

ImageMath 3 ${out_file} ME ${out_file} 20 # Morphological Erosion with radius 30 (focus on the inner brain area)

