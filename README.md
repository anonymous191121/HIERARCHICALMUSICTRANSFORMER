# HIERARCHICALMUSICTRANSFORMER

## How to train


1. Store midi file at midi folder

2. RUN ```python src/midi2txt.py```
(To make midi file to sequence of txt tokens)

3. RUN ```python src/make_group_vocab.py```
(To make group token witch suggest in our paper)

4. ```python -m src.train.chord_train``` , ```python -m src.train.inst_train``` ...
(Train the model)

5. If you want to train particular music for particular attribute (ex. I want to train chord progress with brahms style and train onset style for dvorak), that is also possible for train each model for each txt dataset.

## DEMO Samples


https://www.notion.so/Demo-1d879d40fec3807189a3fe4ec609dcee?pvs=4


(readme fixing...)

### From Scratch Generation

Just start ONLY "BOS" Token for Chord Decocer

Witch means "BOS" Token make entire music


1. 


2. 



### Prompt Generation

Below samples are from 8-measure prompting for all attribute (Chord, Inst, Onset, Duration, Pitch)

Given our model’s architecture, it’s of course possible to pick and prompt only the desired elements for each attribute.

1. 


2. 


3. 


4. 


### Fine Tuning for particular composer attribute

comming soon...


### More sample in .io page

comming soon...
