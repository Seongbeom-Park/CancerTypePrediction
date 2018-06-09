#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Please provide application file"
	exit 1
elif [ ! -f $1 ]; then
	echo "Provided application file does not exist"
	exit 1
fi

bench_path=`realpath $1`
bench_dir=`dirname $bench_path`
bench_file=`basename $bench_path`
bench_name=${bench_file%.*}

mkdir -p results

#echo "cd ${bench_dir}; python ${bench_file}" | qsub -V -v LD_LIBRARY_PATH=/usr/local/cuda/lib64 -N ${bench_name} -o results/${bench_name}.o -e results/${bench_name}.e -l nodes=1:gpus=1:ppn=2
echo "cd ${bench_dir}; python ${bench_file} ${@:2} " | qsub -V -v LD_LIBRARY_PATH=/usr/local/cuda/lib64 -N ${bench_name} -o results/${bench_name}.o -e results/${bench_name}.e -l nodes=1:gpus=1:ppn=2 # s.park changed
