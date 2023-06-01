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
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
