import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def data_cleaning(df):
    df.drop(columns='id', axis=1, inplace=True)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_second"] = df["pickup_datetime"].dt.second
    df["pickup_minute_of_the_day"] = df["pickup_hour"] * 60 + df["pickup_minute"]
    df["pickup_day_week"] =df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month


    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df["dropoff_hour"] = df["dropoff_datetime"].dt.hour
    df["dropoff_minute"] = df["dropoff_datetime"].dt.minute
    df["dropoff_second"] = df["dropoff_datetime"].dt.second
    df["dropoff_minute_of_the_day"] = df["dropoff_hour"] * 60 + df["dropoff_minute"]
    df["dropoff_day_week"] =df["dropoff_datetime"].dt.dayofweek
    df["dropoff_month"] = df["dropoff_datetime"].dt.month

    df.drop(columns=["pickup_datetime", "dropoff_datetime"], axis=1, inplace=True)
    return df

def split_data(df, test_size, seed):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test

def save_data(train, test, out_path):
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(out_path+'/train.csv', index=False)
    test.to_csv(out_path+'/test.csv', index=False)

def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/interim'

    data1 = load_data(data_path)

    data = data_cleaning(data1)
    train_data, test_data = split_data(data, params['test_size'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()