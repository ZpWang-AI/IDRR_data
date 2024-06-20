import pandas as pd

from pathlib import Path as path


def show_csv_info(csv_path):
    df = pd.read_csv(csv_path)
    print('-'*20)
    print(path(csv_path).stem)
    print('total:', len(df))
    print('train:', len(df[df['split']=='train']))
    print('dev  :', len(df[df['split']=='dev']))
    print('test :', len(df[df['split']=='test']))
    print(df.columns)
    