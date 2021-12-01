#! /bin/bash

# Build App

mkdir -p $PWD/app
cd app
mkdir -p $PWD/inputs
mkdir -p $PWD/output

docker create -i --name container1 -v $PWD/inputs:/app/inputs \
				 -v $PWD/output:/app/output \
			         eaubree/project:latest
			         

			     
