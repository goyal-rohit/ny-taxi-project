import pandas as pd
import numpy as np
import sys, pathlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle


def transform_obj(train_df):
    cat_columns = train_df.columns[train_df.dtypes == 'object']
    num_columns = train_df.columns[(train_df.dtypes == 'float') | (train_df.dtypes == 'int')]

    preprocessor = ColumnTransformer(transformers=[
        ('ohe', OneHotEncoder(drop='first', dtype=np.int32), cat_columns),
        ('ss', StandardScaler(), num_columns)
    ], remainder='passthrough'
    )
    return preprocessor


def transform_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    transformed_obj = transform_obj(train_df)

    target_col_name = ['trip_duration']

    input_features_train_df = train_df.drop(target_col_name, axis=1)
    target_feature_train_df = train_df[target_col_name]

    input_features_test_df = test_df.drop(target_col_name, axis=1)
    target_feature_test_df = test_df[target_col_name]

    input_feature_train_arr = transformed_obj.fit_transform(input_features_train_df)
    input_feature_test_arr = transformed_obj.transform(input_features_test_df)

    train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
    test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

    return train_arr, test_arr, transformed_obj


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file_train = sys.argv[1]
    train_path = home_dir.as_posix() + input_file_train
    input_file_test = sys.argv[2]
    test_path = home_dir.as_posix() + input_file_test

    output_path = home_dir.as_posix() + '/data/processed'

    train_arr, test_arr, transformed_obj = transform_data(train_path, test_path)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(output_path+'/trans_obj.pkl','wb') as f:
        pickle.dump(transformed_obj,f)

    #save_obj(output_path, 'trans_obj.pkl', transformed_obj)
    # np.savez(output_path + '/train_arr', train_arr)
    # np.savez(output_path + '/test_arr', test_arr)
    # np.save(output_path + '/train_arr', train_arr)
    # np.save(output_path + '/test_arr', test_arr)
    np.savez_compressed(output_path + '/train_arr', train_arr)
    np.savez_compressed(output_path + '/test_arr', test_arr)


if __name__ == "__main__":
    main()
