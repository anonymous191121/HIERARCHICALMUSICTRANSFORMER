# HIERARCHICALMUSICTRANSFORMER

## How to train


1. Store midi file at midi folder

2. RUN python src/midi2txt.py
(To make midi file to sequence of txt tokens)

3. RUN python src/make_group_vocab.py
(To make group token witch suggest in our paper)

4. python -m src.train.chord_train
(Train the model)