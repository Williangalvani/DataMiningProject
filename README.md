# DataMiningProject


## File preparation

* tokenizer.py: extracts the text features from the provided ".html" files
* fetch_dataset.py: manipulate the files to keep the file tree consistent and immitates the interface used in the other native datasets provided by the scikit-learn API


## Data Analysis

Cases where we are trainning-testing over the different universities:

* document_classification_benchmark.py: evaluation over multiple classifiers, modified to use the CMU datasettet
* grid_search_text_feature_extraction.py: using GridSearch for parameter tuning

* imbalanceNB.py: testing RandomOverSampling to compensate for imbalanced dataset 
* simpleNB.py: simple classification utilizing multinomial naive-bayes
* simpleNBcv.py: simple classification testing if it was possible to separate the dataset into trainning and test using the sklearn module
