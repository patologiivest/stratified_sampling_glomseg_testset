# Balanced weighted sampling

## Description
This repo was created to sample 500 slides from the renal registry whilst balancing for different variables to create a test dataset for glomerular segmentation.

## Features
- Balanced weighted sampling function   
- Sample evaluation metrics
- Sample evaluation visualizations

## Installation
```bash
pip install -r requirements.txt
```

## Usage
For this specific usecase, most likely useless for you. 
```python
python ./scripts/prep_for_stratified_sampling.py
```
Then execute the jupyter notebook with the created data.
Mostly just read through the validation functions for inspiration and adapt them to your use case.

## Requirements
- Python 3.13

## Questions on the way:

It was decided that all immune stains should be grouped and downsampled "as if only one of them existed". HE and HES are grouped and downsampled the same way. Same for masson trichrome and AFOG.
We upsample the slides that have crescent glomeruli because of their rarity.
Retained sampling variables are diagnosis, scanner type, laboratory of origin, stain, and year.
Should we maximize for the number of glomeruli present on the slide? Didn't
Should we filter based on quality? Didn't
