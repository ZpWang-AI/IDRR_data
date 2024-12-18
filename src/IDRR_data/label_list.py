TOP_LEVEL_LABEL_LIST = [
    "Comparison",
    "Contingency",
    "Expansion",
    "Temporal",
]

SEC_LEVEL_LABEL_LIST = {
    "pdtb2": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause",
        "Contingency.Pragmatic cause",
        
        "Expansion.Alternative",
        "Expansion.Conjunction",
        "Expansion.Instantiation",
        "Expansion.List",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous",
        "Temporal.Synchrony",
    ],
    "pdtb3": [
        "Comparison.Concession",
        "Comparison.Contrast",
        # "Comparison.Similarity",  # in CPKD
        
        "Contingency.Cause",
        "Contingency.Cause+Belief",  # not in CPKD
        "Contingency.Condition",
        "Contingency.Purpose",
        
        "Expansion.Conjunction",
        "Expansion.Equivalence",
        "Expansion.Instantiation",
        "Expansion.Level-of-detail",
        "Expansion.Manner",
        "Expansion.Substitution",
        
        "Temporal.Asynchronous",
        "Temporal.Synchronous"
    ],
    "conll": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause.Reason",
        "Contingency.Cause.Result",
        "Contingency.Condition",
        
        "Expansion.Alternative",
        "Expansion.Alternative.Chosen alternative",
        "Expansion.Conjunction",
        "Expansion.Exception",
        "Expansion.Instantiation",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous.Precedence",
        "Temporal.Asynchronous.Succession",
        "Temporal.Synchrony",
    ]
}


if __name__ == '__main__':
    for v in SEC_LEVEL_LABEL_LIST.values():
        if v != sorted(v):
            print(v)
            break
    else:
        print('all sorted')