# project
MNIST classification / docker app

This application was disigned in order to classify hand written digits.

To execute the application : 
	- Docker must be installed on host machine
	
	In file "handWrittenDigits_classification/" :
		- First execute build_app.sh in the current directory.
		- Two directories sould have been created : "inputs/" and "output/". 
		- The inputs directory is where the data to be processed (hand written digit images) should be placed.
		- The output directory is where you will find a json file with the classification result.
		- Once the data is placed in "inputs/" the application can be executed with the file execute.sh
	
In order to execute build_app.sh and execute.sh, open a terminal in the working directory and simply write the command 'bash build_app.sh' and 'bash execute.sh'.


Classifier contains the python code to train the model (MLP) with MNIST data set.
docker_app contains the Dockerfile and the files/directories required to build an image.
MNIST_handwrittendigits contains 10 images of hand written digits extracted from the test set.


Development description :

First the model was trained on MNIST data set and saved. Then the docker image was build with the following organization : an 'app/' directory with two sub-directories 'inputs/' and 'output/'. It also contains, in the 'app/' directory, a file with the trained model and a python script able to load it and to read several images from the inputs folder and write the classification result in the output folder.  
In order to take inputs from the host machine and to deliver an output, two folders (inputs and output) are mounted to the container when it is created. 
The image has been upload on docker hub and a script can be used to simplify the application deployment.
