mitotem
==============================

Analysis of mitochondria in TEM images

Project Organization
------------
```
├── environment.yml    <- The requirements file for reproducing the analysis environment
├── config             <- Project settings and model parameters
│   ├── __init__.py
│   └── settings.py
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks.
│   ├── 1-unet-debug.ipynb
│   ├── 2-dataloader-test.ipynb
│   └── 3-analysis.ipynb
├── README.md          <- The top-level README for developers using this project.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src
│   ├── __init__.py
│   ├── analysis       <- Components used in the analysis pipeline
│   │   └── components.py
│   ├── data           <- Data access related functionality
│   │   └── loaders.py
│   ├── models         <- Scripts to train models and then use trained models to make predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── analysis.py
└── test_environment.py
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
