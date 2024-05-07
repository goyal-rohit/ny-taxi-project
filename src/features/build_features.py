import pathlib
import yaml
import sys
import pandas as pd




def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file_train = sys.argv[1]
    train_path = home_dir.as_posix() + input_file_train
    input_file_test = sys.argv[2]
    test_path = home_dir.as_posix() + input_file_test

    output_path = home_dir.as_posix() + '/data/processed'

 

if __name__ == "__main__":
    main()