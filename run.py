# import jsonl
# a = '{"question": "What position does the player who played for butler cc (ks) play?", ' \
#     '"query_tok": ["SELECT", "position", "WHERE", "school/club", "team", "EQL", "butler", "cc", "(", "ks", ")"], ' \
#     '"query_tok_space": [" ", " ", " ", " ", " ", " ", " ", " ", "", "", ""], ' \
#     '"table_id": "1-10015132-11", ' \
#     '"question_tok_space": [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "", "", " ", "", ""], ' \
#     '"sql": {"agg": 0, "sel": 3, ' \
#     '"conds": [[5, 0, "Butler CC (KS)"]]}, ' \
#     '"phase": 1, ' \
#     '"query": "SELECT position WHERE school/club team EQL butler cc (ks)", ' \
#     '"question_tok": ["what", "position", "does", "the", "player", "who", "played", "for", "butler", "cc", "(", "ks", ")", "play", "?"]}'
#
# print (a)

# # sql_data, table_data, val_sql_data, val_table_data,\
# #         test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = \
# #         load_dataset(0, use_small=USE_SMALL)
# word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
#
# embs = [np.zeros(2,dtype=np.float32) for _ in range(3)]
#
# word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word),
#         use_small=USE_SMALL)
# print "Length of word vocabulary: %d"%len(word_emb)
#
import torch
import torch.nn as nn
print(torch.__version__)
print(torch.cuda.is_available())
# N_word = 300
# B_word = 42
# USE_SMALL = False
# # word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word),
# #         use_small=USE_SMALL)
# # print (word_emb)
rnn = nn.LSTM(10, 20, 2)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))