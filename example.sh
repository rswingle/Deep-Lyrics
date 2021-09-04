#!/usr/bin/env bash

FILE_NAME="rockin"
MODEL_NAME="lstm_regression_model"

# parse web pages to retrieve lyrics, concatenate them and save them locally in a single file
./gather.py --output_file $FILE_NAME.txt --artists "533, 547, 871, 12479, 359, 2011, 445, 601, 420, 6724, 10823, 2538, 101, 119, 994, 611, 12618, 7611, 419, 6612, 6623, 523, 519, 9571, 191, 544, 527"

# create a vocabulary file containing a binary representation for each character
./preprocess.py --input_file $FILE_NAME.txt

# train the LSTM model to fit the lyrics data
./train.py --training_file $FILE_NAME.txt --vocabulary_file $FILE_NAME.vocab --model_name $MODEL_NAME

# generate new lyrics and save them in a file
./sample.py --model_name $MODEL_NAME --vocabulary_file $FILE_NAME.vocab --output_file sample.txt
