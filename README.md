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
part of the notebook involves fine-tuning a Bert model (Distilbert) using the transformers 
(https://huggingface.co/transformers/) library. The only hyperparameter changed was number of epochs. Results:

model |	epoch |	eval_train_loss |	eval_loss |	eval_accHate |	eval_accNoHate | eval_accAll
----- | ----- | --------------- | ----------- | ------------ | --------------- | -----------
0 |	2.0 |	0.150157 |	0.438262 |	0.807531 |	0.836820 |	0.822176
1 |	3.0 |	0.048003 |	0.556381 |	0.799163 |	0.853556 |	0.826360
2 |	4.0 |	0.013872 |	0.701802 |	0.861925 |	0.820084 |	0.841004

We can notice the evaluation loss for 2 epochs is the lowest (this is the most relevant metric). For the second part 
of the notebook I used the pre-trained model (without tuning) to extract embeddings for the entire sentences and pass 
them as the input to the default head classifier (a 2 layer NN). I could have used any other classifier but I chose 
the same default head to make the comparison with the fine-tuning approach fair. There are multiple strategies to select 
embeddings; combinations I tried included:  
* CLS: embedding for [CLS] token.
* LAST_MEAN: mean of all word embeddings from last layer.
* LAST_MAX: max pooling of all word embeddings from last layer (max at each dimension).
* LAST2_MEAN: mean of all word embeddings from second to last layer.
* LAST2_MAX: max pooling of all word embeddings from second to last layer.

It should be noted that, although the [CLS] token is used for a general representation of the sentence for 
classification problems, this representation is meaningless **unless** fine-tuning occurs, otherwise the model wouldn't 
be motivated to to fully funnel the sentence meaning in that single vector. The [CLS] by itself is, therefore, a weak 
embedding; it would be better to test some other aggregations of the other vectors found in the hidden states. You can 
find more information about this topic in the ***bert-as-service*** library documentation 
(https://github.com/hanxiao/bert-as-service#speech_balloon-faq and 
https://hanxiao.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/).

Results for different strategies (all run for 2 epochs and same default parameters):

model |	emb-strategy |	eval_train_loss |	eval_loss |	eval_accHate |	eval_accNoHate |	eval_accAll
----- | ------------ | ---------------- | ----------- | ------------ | --------------- | --------------
0 |	CLS |	0.549668 |	0.574593 |	0.786611 |	0.673640 |	0.730126 
1 |	LAST_MEAN |	0.540307 |	0.559487 |	0.836820 |	0.698745 |	0.767782 
2 |	LAST_MAX |	0.601751 |	0.612578 |	0.828452 |	0.682008 |	0.755230
3 |	LAST2_MEAN |	0.511449 |	0.542598 |	0.811715 |	0.686192 |	0.748954
4 |	LAST2_MAX |	0.585415 |	0.599830 |	0.824268 |	0.686192 |	0.755230

We can see the best strategy by eval_loss is LAST2_MEAN. This is also the default implementation by the 
***bert-as-service*** library.

### Prediction API








