#!/bin/bash

cd tests/data
# Only test_load_pe_samples needs these, and it skips without them, so a failed download is not fatal.
if wget --tries=3 --timeout=30 -O downsampled.tar.gz "https://dcc.ligo.org/public/0182/T2200137/001/O3bPE_downsampled.tar.gz"; then
    tar -xvzf downsampled.tar.gz
else
    rm -f downsampled.tar.gz
    echo "::warning::could not download the O3b PE samples from dcc.ligo.org; test_load_pe_samples will be skipped"
fi
wget --tries=3 --timeout=30 -O injections.h5 "https://zenodo.org/record/5546676/files/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
cd ../../
