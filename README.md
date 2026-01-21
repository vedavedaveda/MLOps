# Group 39 - MLOps Project

**Overall goal:** The goal of the project is to perform binary image classification of artwork images in order to determine whether an image is AI-generated or human generated.

**Framework:** We will use pytorch/pytorch lightning for the deep learning framework, as well as Docker to containerise our solution. We will probably add more to our framework once we start working.

**Data:** We've found a database on Kaggle of artwork labelled as either human-made or AI-generated. The dataset contains 975 artworks, where about half of which are AI-generated. [The database can be downloaded here.](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data) Since we may only be able to work with a smaller subset of the data (due to compute limitations), we may need to consider ways to mitigate overfitting. The dataset was published around a year ago, so the images are relatively representative of current AI image generation capabilities.

**Models:** We are planning to use a simple CNN model to classify the images as either AI-generated or human-made. As mentioned above, we may need to include strategies to mitigate overfitting, but we will make adjustments to the pipeline once we've completed the simple implementation.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Hyperparameter Sweep
https://wandb.ai/frejaj9-danmarks-tekniske-universitet-dtu/MLOps-src_ml_ops_project?nw=nwuserfrejaj9

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

Running AI classifier frontend and backend as microservices with docker

## Notes for us

### DVC/Cloud stuff:
```uv run dvc add data```
```uv run dvc push --no-run-cache```
```uv run dvc pull --no-run-cache```

### Running AI classifier frontend and backend as microservices with docker

```docker build -t backend:latest -f backend.dockerfile .```
```docker run --rm -p 8001:8080 backend:latest```

Open new terminal and type following commands:

```docker build -t frontend:latest -f frontend.dockerfile .```
```docker run --rm -p 8501:8501 -e "BACKEND=http://host.docker.internal:8001" frontend:latest```

### Running pre-commit rules

Should be done prior to every time you commit
```uv run pre-commit run --all-files```
