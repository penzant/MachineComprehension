#!/bin/bash
# data/Statements - MCTest data, but with questions and answers converted into statements, using the rules employed in a web-based question answering system

# data/MCTest - 660 reading comprehension tests (split into MC160 and MC500)

    from logistic_sgd import LogisticRegression, load_data
  File "logistic_sgd.py", line 48, in <module>
    from cis.deep.utils.theano import debug_print
ImportError: No module named cis.deep.utils.theano

# Import data from http://research.microsoft.com/en-us/um/redmond/projects/mctest/data.html.
cd data
wget http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/
unzip MCTest.zip
rm MCTest.zip
wget http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/MCTestAnswers.zip
unzip MCTestAnswers.zip
mv MCTestAnswers/* MCTest
rmdir MCTestAnswers
rm MCTestAnswers.zip
wget http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/Statements.zip
unzip Statements.zip
rm Statements.zip
wget http://research.microsoft.com/en-us/um/redmond/projects/mctest/data/stopwords.txt

# Prepare the data.

python preprocess_mctest.py

# Train the model.
#python train_arc1.py
python train_entailment_v3.py
