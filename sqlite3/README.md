## Simple SQLite3 Exercise

These scripts serve as an exercise to get familiar with debugging, SQLite3, some builtin Python modules, and general function architecture, for example, error handling.

## Set Up

No additional downloads/installations necessary. SQLite3 will run straight out of the box. 

To start the exercise, simply type in the commandline (in the same directory as where the scripts are):

```
python3 sql_exercise_main.py
```

Error messages will be presented, indicating at which part in the script the first error was identified. 

To fix the errors, open the following file in an editor:

sql_exercise_broken_functions.py

And fix the functions by following the directions at the top. Basically, all you have to do is insert the functions or variables listed at the top into the functions.

## Goals of Script:

This is a script for learners of Python to practice the following:

1) working with SQLite3
* How to connect to a database
* How to execute commands with SQLite3
* How to save changes and close the database

2) debugging of scripts
* Get an error message and see what needs to be fixed.

3) collecting user input and handle strings in Python
* How to check user input (what did the user type in)
* use string modules of Python
* How to use placeholders in strings to insert variables

4) converting types of obects
* For example, if the object is of type string, how to change it to an integer.

5) error and exception handling
* try, except and finally statments
* raising an exception
