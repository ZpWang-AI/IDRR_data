import pandas as pd


def show_df_info(df:pd.DataFrame):
    print('columns:', (df.columns))
    print('splits :', set(df['split']))
    print('-'*20)
    print('total:', len(df))
    print('train:', len(df[df['split']=='train']))
    print('dev  :', len(df[df['split']=='dev']))
    print('test :', len(df[df['split']=='test']))
    blind = len(df[df['split']=='blind-test'])
    if blind:
        print('blind:', blind)
    print('sum  :', sum(
            len(df[df['split']==split]) 
            for split in 'train dev test blind-test'.split()
        )
    )


if __name__ == '__main__':
    df = pd.read_csv(r'D:\ZpWang\Projects\02.01-IDRR_data\data\used\conll.p1.csv')
    df = pd.read_csv(r'D:\ZpWang\Projects\02.01-IDRR_data\data\used\conll_test.p1.csv')
    df = pd.read_csv(r'D:\ZpWang\Projects\02.01-IDRR_data\data\used\conll_blind_test.p1.csv')
    # df = pd.read_csv(r'D:\ZpWang\Projects\02.01-IDRR_data\data\used\pdtb3_test.p1.csv')
    show_df_info(df)
    