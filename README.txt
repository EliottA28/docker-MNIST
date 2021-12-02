# project
MNIST classification / docker app

This application was disigned in order to classify hand written digits.

To build the application : 
	- Docker must be installed on host machine
	
	- First execute build_app.sh in your working directory
	- A new directory should have been created and named "handWrittenDigits_classification/" as well as two sub-directories : "inputs/" and "output/". 
	- The inputs directory is where the data to be processed (hand written digit images) should be placed.
	- The output directory is where you will find a json file with the classification result after application execution.
	- Once the data is placed in "inputs/" the application can be executed with the file execute.sh
	
In order to execute build_app.sh and execute.sh, open a terminal in working directory and simply write the command 'bash build_app.sh' and 'bash execute.sh'.
