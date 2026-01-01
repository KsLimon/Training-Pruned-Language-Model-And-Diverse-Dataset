# Bangla Word2Vec Pre-trained Model
This word2vec model trained with `Bangla wikipedia` dump datasets.

## Corpus Details
- Data size: 417.3MB
- Total Sentences: 1654772
- Total Tokens: 26446380
- Total Unique Tokens: 1001170

## Training Details
- vector_size = 100
- window = 5
- min_count = 5
- sample=1e-3
- cbow_mean=1
- epoch = 10

## Evaluation Results
We evaluate with a word list, result was impressive. check `evaluation` folder.