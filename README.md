# Hate speech prediction

This project develops a model for binary classification on hate speech sentences. To accomplish this I am using the 
Bert (Bidirectional Encoder Representations from Transformers) language model. Components of the project:

- A notebook with EDA on the dataset found in *notebooks/EDA.ipynb*.
- A notebook with Bert modeling approaches (the meat of the project) found in *notebooks/bert.ipynb*.
- An API for prediction around the final model.

### Background

The baseline work is described in the following paper:
https://arxiv.org/pdf/1809.04444.pdf. This previous work includes a dataset scraped from a white
nationalist forum (https://www.stormfront.org/forum/), some data exploration, and some baseline models. It also 
includes a description of the annotation process.

Classifying hate speech is not a straight forward task. What is considered hate speech might be very subjective. 
Researches describe the issues encountered when annotating/labeling the dataset. They chose to label text at
sentence level because: “Sentence-level annotation allows to work with the minimum unit containing
hate speech and reduce noise introduced by other sentences that are clean” (page 3). The authors
mention that there is no consensus as to what exactly hate speech is, so they set up a criteria/guidelines
for annotators to accurately label the sentences. According to this criteria hate speech must include (page 3):

- a deliberate attack
- directed towards a specific group of people
- motivated by aspects of the group’s identity.

Even with these guidelines there was multiple differences among annotators. Some questions that might arise include 
(some are mentioned in the paper): 

- can a factual statement constitute an attack?
- can a single individual constitute an entire group if the offense can be generalize?
- when does a trait becomes associated with an entire group's identity? Couldn't a trait be mentioned only in reference 
of a single individual or perhaps a subgroup?  

Some of these differences where discussed in order to homogenize the results as much as possible. 

### Dataset

Dataset can be found here: https://github.com/Vicomtech/hate-speech-dataset.git. It contains the original train/test 
split. There are in total 2392 observations with 1914 and 478 instances for training and testing respectively. Classes 
are balanced in both sets. Dataset head:

file_id	text |	gSet |	label | set | label
------------ | ----- | ------ | --- | -----
0	| 14103132_1 |	Five is about what I 've found also . |	train |	noHate
1	| 14663016_2 |	Mexicans have plenty of open space in their co... |	train |	hate
2	| 13860974_1 |	I didnt know that bar was owned by a negro i w... |	train |	hate
3	| 30484029_2 |	If I had one it would 've probably made the li... |	train |	noHate
4   | 13864997_3 |	Most White Western Women have the racial sense... |	train |	hate

Some of the top keywords by log-likelihood test found in out group of interest ***hate*** include: 

keyword | likelihood
------- | ----------
black | 58.513906
jews | 54.982564
negro | 40.754824
ape | 33.393051
race | 24.915494
scum | 24.367472

For more data exploration go to *notebooks/EDA.ipynb* (data is downloaded automatically in the notebook 
so you don't need to downloaded yourself).

### Bert Model

My main goal in the data modeling phase was to compare 2 approaches: fine-tuning vs. embeddings. The first 
part of the notebook involves 