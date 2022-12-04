# Brain Age Prediction
Mini-Project, Medical Image Computing Course 2020/21, King's College London

In this project, we create an end-to-end tool to estimate patient age according to volumetric measurements of brain MRI.
The project consists of three parts: generation of prior maps, segmentation of brain volumes and fitting the model
to estimate patient age.
The dataset of 10 segmented brain MRI volumes is used to create the prior maps for CSF, white matter and grey matter.
The second dataset includes 20 unsegmented brain MRI volumes with labels describing the patient's age.

In order to generate prior maps, the segmented volumes are registered to an average to produce transformations to use
for segmentations maps. The registered segmentation maps are averaged across the patients to estimate the prior maps.
The average volume is registered to unsegmented volumes, producing transformations to match the priors
to the unsegmented volumes.

The segmentation of the 20 brain MRI volumes is performed using Gaussian Mixture Model (GMM).
The implemented GMM can use prior maps and Markov random field (MRF) regularisation.
As a result, the GMM produces segmentation estimations to 20 brain MRI volumes.

The prediction of patient age is performed according to CSF, wm and gm volume sizes.
While many possible solutions are available, we performed a grid search of hyperparameters for polynomial and ridge
regressions. 
The best model was used to deliver final estimations of patient age.

## Getting Started

### Setting up the project
Clone the repo to your local machine.
```python
git clone https://github.com/denproc/brainage.git
cd brainage
```
[Optional] Create a virtual environment.
```python
python -m venv ./venv
source venv/bin/activate
```
Install required dependencies.
```python
pip install -r requirements.txt
```
### Run Scripts
Run scripts to use the tool step by step.
```python
python registration.py  # Generate average template and transformations to its space
python priors.py  # Generate priors based on available segmentations
python register_priors.py  # Generate transformations from unsegmented cases to the template 
python priors_for_unsegmented.py  # Transform priors to unsegmented cases
python segmentation.py  # Run segmentation using GMM
```

### Explore Notebooks
Use `jupyter` notebooks  (`./notebooks`) to play and visualise registration, segmentation and brain age prediction.


## Contacts
Denis Prokopenko - [@denproc](https://github.com/denproc) - `d.prokopenko@outlook.com` 
