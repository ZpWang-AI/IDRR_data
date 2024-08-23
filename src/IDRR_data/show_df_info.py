import pandas as pd


def show_df_info(df:pd.DataFrame):
    print('total:', len(df))
    print('train:', len(df[df['split']=='train']))
    print('dev  :', len(df[df['split']=='dev']))
    print('test :', len(df[df['split']=='test']))
    print('sum  :', sum(
            len(df[df['split']==split]) 
            for split in 'train dev test'.split()
        )
    )
    print(df.columns)
    print('-'*20)
    