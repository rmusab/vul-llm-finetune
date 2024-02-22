# LineVul

## Docker run 

```
docker run -it -v /hdd/predictions:/home/ma-user/linevul/predictions -v /hdd/vul-detection/LineVul/data/:/home/ma-user/linevul/data --user 1000:100 -e DATA_PATH=/home/ma-user/linevul/data/cwe-502_without_random_without_is_var.tar.gz -e OUTPUT_PATH=/home/ma-user/linevul/predictions -e EPOCHS=6 -e TRAIN_BATCH_SIZE=64 -e EVAL_BATCH_SIZE=64 docker_linevul:latest
```

Parameters to run docker image locally:

1. `/hdd/predictions`: This is the folder where the model predictions for the test will be stored as text files.
2. `/hdd/vul-detection/LineVul/data/`: This is the folder where the compressed data file (tar.gz) containing the train, test and validation sets is located.
3. DATA_PATH: This is the local path where the data file (tar.gz) will be extracted and loaded by the model. The data file should have the same name and structure as described above.
4. –user 1000:100: This is the user and group ID that will be used to run the docker container. The default user for Roma is Ubuntu, which has the user ID 1000 and the group ID 100. This argument ensures that the files created by the container have the same permissions as the host system.
5. OUTPUT_PATH: This is the local path where the model predictions for the test and validation sets will be copied from the container to the host system. 
6. EPOCHS=6: This is the number of epochs that the model will train for. 
7. TRAIN_BATCH_SIZE=64: This is the number of examples that the model will process in one iteration of training. 
8. EVAL_BATCH_SIZE=64: This is the number of examples that the model will process in one iteration of evaluation or testing. 
9. CHOOSE_BEST_THRESH=True/False: Should optimize threshold for validation to use then it in testing procedure 
10. MODEL_NAME: model which will be loaded instead of microsoft/codebert. It must be located in `/home/ma-user/linevul/`. E.g., contrabert has been already put inside docker image, so you can set `MODEL_NAME=contrabert`.