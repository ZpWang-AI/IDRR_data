import pandas as pd
import numpy as np
import json
import os
import re

from typing import *
from IDRR_data.label_list import TOP_LEVEL_LABEL_LIST, SEC_LEVEL_LABEL_LIST
from IDRR_data.ans_word_map import ANS_WORD_LIST, ANS_LABEL_LIST, SUBTYPE_LABEL2ANS_WORD


def ans_words2token_id(ans_words, tokenizer):
    import transformers
    vocab = tokenizer.get_vocab()
    if isinstance(tokenizer, transformers.RobertaTokenizer) or \
        isinstance(tokenizer, transformers.RobertaTokenizerFast):
        def ans_word_tokenizer(word:str):
            return vocab['Ġ'+word.strip()]
    elif isinstance(tokenizer, transformers.BertTokenizer) or \
        isinstance(tokenizer, transformers.BertTokenizerFast):
        def ans_word_tokenizer(word:str):
            return vocab[word.strip()]
    else:
        raise Exception('wrong type of tokenizer')
    
    return list(map(ans_word_tokenizer, ans_words))


"""
init columns:
    'arg1', 'arg2', 'conn1', 'conn2', 
    'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2',
    'relation', 'split', 'data_id'
process:
    connXsenseY -> labelXY, labelXYid
    connX -> ans_wordX, ans_wordXid
    relation: filter
    split: filter
processed columns:
    'index', 'arg1', 'arg2', 'conn1', 'conn2', 
    'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2', 
    'relation', 'split', 'data_id', 
    'label11', 'label11id', 'label12', 'label12id', 
    'label21', 'label21id', 'label22', 'label22id', 
    'ans_word1', 'ans_word1id', 'ans_word2', 'ans_word2id'
"""
class IDRRDataFrames:
    def __init__(
        self,
        data_name:Literal['pdtb2', 'pdtb3', 'conll']=None,
        data_level:Literal['top', 'second', 'raw']='raw',
        data_relation:Literal['Implicit', 'Explicit', 'All']='Implicit',
        data_path:Optional[str]=None,
    ) -> None:
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        assert data_level in ['top', 'second', 'raw']
        assert data_relation in ['Implicit', 'Explicit', 'All']
        self.data_name = data_name
        self.data_level = data_level
        self.data_relation = data_relation 
        self.data_path = data_path
    
        self.df = pd.DataFrame()
        if data_path:
            self.load_df(data_path)
    
    def load_df(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path, low_memory=False)
    
    @property
    def json(self):
        return {
            'data_name': self.data_name,
            'data_level': self.data_level,
            'data_relation': self.data_relation,
            'data_path': (self.data_path if self.data_path else ''),
        }
        
    def __repr__(self):
        return f'{self.data_name}_{self.data_level}_{self.data_relation}'
        
    # =================================================================
    # Dataframe
    # =================================================================
    
    def get_dataframe(self, split=Literal['train', 'dev', 'test', 'all', 'raw']) -> pd.DataFrame:
        df = self.df.copy(deep=True)
        # relation
        if self.data_relation != 'All':
            df = df[df['relation']==self.data_relation]
        # split
        assert split in ['train', 'dev', 'test', 'all', 'raw']
        if split == 'raw':
            pass
        elif split == 'all':
            df = df[~pd.isna(df['split'])]
        else:
            df = df[df['split']==split]
        # ans word, label
        if self.data_name and self.data_level != 'raw':
            df = self.process_df_sense(df)
            df = self.process_df_conn(df)
            df = df[pd.notna(df['label11'])]
            df = df[pd.notna(df['ans_word1'])]
        df.reset_index(inplace=True)
        return df
    
    @property
    def train_df(self) -> pd.DataFrame:
        return self.get_dataframe('train')
    
    @property
    def dev_df(self) -> pd.DataFrame:
        return self.get_dataframe('dev')
        
    @property
    def test_df(self) -> pd.DataFrame:
        return self.get_dataframe('test')
    
    @property
    def all_df(self) -> pd.DataFrame:
        return self.get_dataframe('all')
            
    # =================================================================
    # Label
    # =================================================================

    @property
    def label_list(self) -> List[str]:
        if self.data_level == 'top':
            label_list = TOP_LEVEL_LABEL_LIST
        elif self.data_level == 'second':
            label_list = SEC_LEVEL_LABEL_LIST[self.data_name]
        else:
            raise Exception('wrong data_level')
        return label_list     
           
    def label_to_id(self, label):
        return self.label_list.index(label)
    
    def id_to_label(self, lid):
        return self.label_list[lid]
    
    def process_sense(
        self, sense:str,
        label_list=None, 
        irrelevent_sense=pd.NA,
    ) -> Tuple[str, int]:
        """
        match the longest label
        return: label, lid
        """
        if pd.isna(sense):
            return (irrelevent_sense,)*2 

        if not label_list:
            label_list = self.label_list
        
        res_label = max(
            label_list,
            key=lambda label: (
                sense.startswith(label),
                len(label)
            )    
        )
        res_lid = label_list.index(res_label)
        if sense.startswith(res_label):
            return res_label, res_lid
        else:
            return (irrelevent_sense,)*2
        
    def process_df_sense(self, df:pd.DataFrame):
        label_list = self.label_list
        
        for x,y in '11 12 21 22'.split():
            sense_key = f'conn{x}sense{y}'
            label_key = f'label{x}{y}'
            label_values, lid_values = [], []
            for sense in df[sense_key]:
                label, lid = self.process_sense(
                    sense=sense, label_list=label_list, irrelevent_sense=pd.NA,
                )
                label_values.append(label)
                lid_values.append(lid)
            df[label_key] = label_values
            df[label_key+'id'] = lid_values
        return df
    
    # =================================================================
    # Ans word
    # =================================================================
    
    @property
    def ans_word_list(self) -> list:
        return ANS_WORD_LIST[self.data_name]
    
    def get_ans_word_token_id_list(self, tokenizer) -> list:
        return ans_words2token_id(ans_words=self.ans_word_list, tokenizer=tokenizer)
    
    @property
    def ans_label_list(self) -> list:
        return ANS_LABEL_LIST[self.data_name][self.data_level]
    
    @property
    def ans_lid_list(self) -> list:
        return list(map(self.label_to_id, self.ans_label_list))
        
    @property
    def subtype_label2ans_word(self) -> dict:
        return SUBTYPE_LABEL2ANS_WORD[self.data_name]
    
    def ans_word_to_id(self, ans_word):
        return self.ans_word_list.index(ans_word)
    
    def id_to_ans_word(self, awid):
        return self.ans_word_list[awid]
    
    def process_conn(
        self, conn:str, sense:str,
        ans_word_list:list=None, 
        subtype_label2ans_word:dict=None,
        irrelevent_conn=pd.NA,
    ) -> Tuple[str, int]:
        if not ans_word_list:
            ans_word_list = self.ans_word_list
        if not subtype_label2ans_word:
            subtype_label2ans_word = self.subtype_label2ans_word
        
        if pd.notna(conn) and conn in ans_word_list:
            return conn, self.ans_word_to_id(conn)
        
        if pd.notna(sense):
            if sense not in subtype_label2ans_word:
                sense = sense.split('.')[0]
                # return (irrelevent_conn,)*2  # CPKD
            conn = subtype_label2ans_word[sense]
            return conn, self.ans_word_to_id(conn)
        else:
            return (irrelevent_conn,)*2
    
    def process_df_conn(self, df:pd.DataFrame):
        ans_word_list = self.ans_word_list
        subtype_label2ans_word = self.subtype_label2ans_word
        
        for x in '12':
            conn_key = f'conn{x}'
            ans_word_key = f'ans_word{x}'
            ans_word_values, awid_values = [], []
            for conn, sense in zip(df[conn_key], df[conn_key+'sense1']):
                ans_word, awid = self.process_conn(
                    conn=conn, sense=sense, ans_word_list=ans_word_list,
                    subtype_label2ans_word=subtype_label2ans_word,
                    irrelevent_conn=pd.NA,
                )
                ans_word_values.append(ans_word)
                awid_values.append(awid)
            df[ans_word_key] = ans_word_values
            df[ans_word_key+'id'] = awid_values
        return df
    