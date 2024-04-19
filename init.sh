#!/bin/bash

cd 3rdparty; cd cutlass-extension; mkdir build; cd build; cmake ..

if [ "$(nproc)" -le 16 ]; then
    threads=$num_processors
else
    threads=16
fi
make -j"$threads"

cd ../../../; pip3 install -r requirements.txt; python setup_cuda.py install

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/3rdparty/cutlass-extension/build/lib"
