Observations
=============

A new folder named with the current timestamp is created on successful execution of the program.
The folder contains the following files:

1. command.txt
--------------

The python command that created this folder

2. repFieldsXXX.png
--------------------

The receptive fields saved iteratively. The last 3 digits in the filename indicate the iteration number


3. repFields.png
-----------------

The final receptive field after training

4. test_err.txt
----------------

The error on the test set after each iteration. First column indicates iteration number; and second the test error

5. train_err.txt
----------------

The error on the train set after each iteration. First column indicates iteration number; and second the train error

6. validation_err.txt
----------------------

The error on the validation set after each iteration. First column indicates iteration number; and second the validation error

7. error.png
-------------

The plot of the above errors




