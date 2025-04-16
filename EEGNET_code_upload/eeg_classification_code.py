

for i in (1, 3):
    from test import classification_code_subs
    #sub, epochnum, fold
    classification_code_subs(i, 3, 2)
    
    del classification_code_subs
