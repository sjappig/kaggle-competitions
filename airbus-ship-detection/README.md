# Usage

* Copy train JPEGs to *train/* and *train_ship_segmentations.csv* in this directory

* Create virtual environment: `virtualenv --python=python3.6 --system-site-packages venv`

* Activate it: `source venv/bin/activate`

* Install dependencies: `pip install -r requirements.txt`

* Train model: `python -m work.train`

* View few randomly chosen samples: `python -m work.view`

