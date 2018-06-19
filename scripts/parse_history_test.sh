#!/bin/bash

curr_step=0
declare -A result
result['BRCA[0]']=0
result['COADREAD[1]']=0
result['GBM[2]']=0
result['LUAD[3]']=0
result['OV[4]']=0
result['UCEC[5]']=0
declare -A total
total['BRCA[0]']=0
total['COADREAD[1]']=0
total['GBM[2]']=0
total['LUAD[3]']=0
total['OV[4]']=0
total['UCEC[5]']=0

while read -r line ; do
	IFS_back=$IFS
	IFS=','
	data=($line)
	IFS=$IFS_back

	if [ ${#data[@]} != 5 ]; then
		echo "$curr_step,${result['BRCA[0]']},${result['COADREAD[1]']},${result['GBM[2]']},${result['LUAD[3]']},${result['OV[4]']},${result['UCEC[5]']},${total['BRCA[0]']},${total['COADREAD[1]']},${total['GBM[2]']},${total['LUAD[3]']},${total['OV[4]']},${total['UCEC[5]']}"

		curr_step=$(($curr_step + 1))
		result['BRCA[0]']=0
		result['COADREAD[1]']=0
		result['GBM[2]']=0
		result['LUAD[3]']=0
		result['OV[4]']=0
		result['UCEC[5]']=0
		total['BRCA[0]']=0
		total['COADREAD[1]']=0
		total['GBM[2]']=0
		total['LUAD[3]']=0
		total['OV[4]']=0
		total['UCEC[5]']=0
		continue
	fi

	step=${data[0]}
	progress=${data[1]}
	sample=${data[2]}
	expected=${data[3]}
	predicted=${data[4]}

	
	total["$expected"]=$((total["$expected"] + 1))
	if [ "$expected" == "$predicted" ]; then
		result["$expected"]=$((result["$expected"] + 1))
		#echo result[$expected]=$((result[$expected] + 1))
	fi
done < "history_test.csv"
