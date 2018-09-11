# Airbus Ship Detection Challenge

## Usage

### Create virtual environment (optional)

* Create virtual environment: `virtualenv --python=python3.6 --system-site-packages venv`

* Activate it: `source venv/bin/activate`

* Install dependencies: `pip install -r requirements.txt`


### Train model

* Copy train JPEGs to *train/* and *train_ship_segmentations.csv* in this directory

* Train a model: `python -m work.train --samples 64 --epochs 4 --batch 16 --val 0.2 --trained_model my_model.h5` (`--help` to see instructions)


### Create submission

* Copy test JPEGs to *test/* and *sample_submission.csv* in this directory

* Test the trained model: `python -m work.test --samples 64 --batch 16 --trained_model my_model.h5` (`--help` to see instructions)


### Misc

* View few randomly chosen training samples with prediction: `python -m work.view --trained_model my_model.h5`

* Run unit tests: `python -m unittest`

* Run pylint: `python -m pylint work`

