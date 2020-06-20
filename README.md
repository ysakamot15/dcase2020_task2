# dcase2020_task2
This repository is an our solution for DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring".

http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

These scripts divert part of the baseline system of this task.

https://github.com/y-kawagu/dcase2020_task2_baseline

# description
Main script of this system is `train_test.py`.
You need to edit `dcase_task2.yaml` before run the script.
At least, edit the `dev_directory` and `eval_directory` items to specify the dataset directory correctly.
And when you want to get the result of develop dataset, run the follow command. 
```
$ python3 train_test.py -d
```
get the result of evalution dataset, run the follow command.
```
$ python3 train_test.py -e
```