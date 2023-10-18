#!/bin/bash

export ROOT_PATH='/media/khoa-ys/Personal/Projects/Football Analysis/Football-analysis/data'

#echo ${ROOT_PATH}

export VID_PATH=${ROOT_PATH}/'videos'
export IMAGE_PATH=${ROOT_PATH}/'images'

#echo ${VID_PATH}
#echo ${IMAGE_PATH}

cd "$VID_PATH"

for File in *
do
	echo $File
	export OUTPUT_PATH=${IMAGE_PATH}/"${File%%.*}"
	echo $OUTPUT_PATH
	mkdir "${OUTPUT_PATH}"
	ffmpeg -i "${File}" -vf "select=not(mod(n\,60))" -vsync vfr "${OUTPUT_PATH}"/%12d.png
done

