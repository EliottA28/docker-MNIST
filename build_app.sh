#! /bin/bash

# Build App

mkdir $PWD/handWrittenDigits_classification

cd handWrittenDigits_classification

mkdir -p $PWD/inputs
mkdir -p $PWD/output

touch execute.sh
echo -n "docker start " > execute.sh

docker create -i -v $PWD/inputs:/app/inputs \
				 -v $PWD/output:/app/output \
			         eaubree/project:latest >> execute.sh