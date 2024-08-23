import pandas as pd
import re
# import transformers

from typing import *


class PromptFiller:
    def __init__(
        self, df:pd.DataFrame, prompt, 
        ignore=(), tokenizer=None,
    ) -> None:
        self.df = df
        self.prompt = prompt
        self.ignore = ignore
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for index, row in self.df.iterrows():
            res = self.fill_prompt(
                row=row, 
                prompt=self.prompt, 
                ignore=self.ignore, 
                tokenizer=self.tokenizer
            )
            yield res            
    
    @property
    def list(self):
        return list(self)
    
    @classmethod
    def fill_prompt(
        cls, 
        row:Union[pd.Series, dict],
        prompt, 
        ignore=(), 
        tokenizer=None,
    ):
        if isinstance(prompt, str):
            def replace_func(blank:re.Match):
                blank = blank.group()[1:-1]
                if ignore and blank in ignore:
                    return ''
                try:
                    blank_val = row.__getattr__(blank)
                    return str(blank_val) if pd.notna(blank_val) else ''
                except:
                    raise Exception(f'row don\'t have blank(attr): {blank}')  
            
            prompt = re.sub(r'\{[^{}]*\}', replace_func, prompt)
            if '<mask>' in prompt:
                # assert tokenizer.mask_token, f'{type(tokenizer)} has no mask_token'
                prompt = prompt.replace('<mask>', tokenizer.mask_token)
            return prompt

        elif isinstance(prompt, dict):
            return {
                k: cls.fill_prompt(row=row, prompt=v, ignore=ignore, tokenizer=tokenizer)
                for k,v in prompt.items()
            }
        elif isinstance(prompt, (list, tuple)):
            return [
                cls.fill_prompt(row=row, prompt=p, ignore=ignore, tokenizer=tokenizer)
                for p in prompt
            ]
        else:
            raise Exception('wrong type of prompt')
            
    
if __name__ == '__main__':
    from .dataframes import IDRRDataFrames
    pdtb2_df = IDRRDataFrames(
        data_name='pdtb2',
        label_level='level2',
        relation='Implicit',
        data_path=r'/public/home/hongy/zpwang/LLM_Reasoning/data/used/pdtb2.p1.csv',
    )
    df = pdtb2_df.train_df
    df['empty'] = pd.NA
    filler = PromptFiller(
        df, prompt='{conn1sense1}  {relation} >>{empty}<<'
    )
    for p, fp in enumerate(list(filler)):
        print(fp)
        if p == 3:
            break
        