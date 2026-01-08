# ml_ops_project

**Overall goal:** The goal of the project is to perform binary image classification of artwork images in order to determine whether an image is AI-generated or human generated.

**Framework:** We will use pytorch for the deep learning framework, as well as Docker to containerise our solution. We will probably add more to our framework once we start working.

**Data:** We've found a database on Kaggle of artwork labelled as either human-made or AI-generated. The dataset contains 975 artworks, where about half of which are AI-generated. [The database can be downloaded here.](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art/data) Since the dataset is relatively small for a deep learning project, we will need to consider ways to mitigate overfitting. The dataset was published within the last few months, so the images are representative of current AI image generation capabilities.

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

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
