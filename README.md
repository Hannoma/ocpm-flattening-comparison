Predictive Process Mining
==============================

Traditional process mining techniques work with event data that assume one object (traditionally
called case identifier) per event. Each object represents one instance of the
process. In reality, events are often associated with multiple objects, expressing an interaction
or dependency of those objects. Therefore, the assumption of one object per event
is unrealistic, and object-centric event logs have been proposed, where each event relates
to different objects, which also can be related among themselves. Predictive process monitoring
techniques are concerned with predicting the future of a running (uncompleted)
execution of a business process. Predicting how a business process will unfold precisely
can be valuable in several domains. Since most predictive process monitoring techniques
work on traditional flat event logs, we need to flatten the event data. Flattening describes
the process that introduces a single-case notion turning an object-centric event log into a
traditional flat event log. There are multiple choices possible of which object type(s) to
choose as case notion, leading to different views that are disconnected or inconsistent. The
resulting flattened log may suffer from deficiency, convergence, or divergence. Or in other
words, flattening may lead to a mutation of the event data: removing events, replicating
events, or falsifying precedence constraints. Those three issues introduced by flattening
event logs can impact feature values and encodings of them. This influence propagates to
their encoded form used as training input for the predictive models. As a result, the accuracy
of predictive results might be affected. However, it remains unclear how extensive this
influence is and what improvements can be achieved using a thoroughly object-centric predictive
process monitoring pipeline that does not flatten the data beforehand and avoids
the associated quality problems altogether.
To isolate the impact of flattening, we develop an experimental framework that only varies
in the event log used as input. It is based on a general framework for extracting and encoding
features from object-centric event data. By adding a flattening step in the beginning,
the framework allows us to compare the results using an object-centric event log to a
flattened version of it.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, object-centric event logs in jsonocel format.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to convert or flatten event logs
    │   │   ├── clean_up.py <- Script to convert a raw dataset into an object-centric event log and preprocess it
    │   │   ├── convert.py  <- Script to convert a pandas dataframe into an object-centric event log
    │   │   └── flatten.py  <- Script to flatten an object-centric event log based on a compound or single object type
    │   │
    │   ├── encoding       <- Scripts to encode features
    │   │   ├── graph.py    <- Script to encode features as feature graph prefixes
    │   │   ├── sequential.py  <- Script to encode features as feature sequence prefixes
    │   │   └── tabular.py   <- Script to encode features as a tabular representation
    │   │
    │   ├── features       <- Scripts to extract features from object-centric event logs
    │   │   └── build_features.py
    │   │
    │   ├── helpers        <- Utility functions used in the project
    │   │   ├── caching.py <- Script to cache results of functions
    │   │   ├── dataset.py  <- Script to work with defined datasets
    │   │   └── lookup_table.py <- A hashable lookup table
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── gnn.py      <- Script to train a graph neural network
    │   │   ├── lstm.py     <- Script to train a long short-term memory network
    │   │   └── train_model      <- Script to train simple regression models and graph embedding models, and to make predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   ├── running_example.py <- Create visualizations for a running example
    │   │   └── visualizations.py <- Create visualizations for the results of the experiments


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
