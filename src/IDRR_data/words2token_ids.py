def words2token_ids(words, tokenizer):
    import transformers
    vocab = tokenizer.get_vocab()
    if isinstance(tokenizer, transformers.RobertaTokenizer) or \
        isinstance(tokenizer, transformers.RobertaTokenizerFast):
        def word_tokenizer(word:str):
            return vocab['Ġ'+word.strip()]
    elif isinstance(tokenizer, transformers.BertTokenizer) or \
        isinstance(tokenizer, transformers.BertTokenizerFast):
        def word_tokenizer(word:str):
            return vocab[word.strip()]
    else:
        raise Exception('wrong type of tokenizer')
    
    return list(map(word_tokenizer, words))