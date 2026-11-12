# PET/CT Lesion Segmentation — N03

## Project
A semi-automatic PET/CT lesion segmentation tool using SUV thresholds and a user-defined region of interest.

## Members
- 20224386
- 20224388
- 20233973

All members contribute to dataset search, slide preparation and code testing.

## Files
- [Progress Test 2 report](report/kiem_tra_lan_2.md)
- [Application source code](src/segmentation_14.py)
- [Test script](src/test_segmentation.py)
- [Technical test results](results/test_results.json)
- [Data checksums](results/data_checksums.json)

## Run the application
Install Python and the required packages:

    pip install numpy SimpleITK connected-components-3d opencv-python PyQt5 Pillow scipy

From this folder, run:

    python src/segmentation_14.py

Select the CT image and the SUV image.
The application does not convert raw PET values to SUV.
Draw an initial region before using Refine.

## Current limitations
Independent baseline-versus-proposed Dice evaluation is not complete.
The existing-mask comparison is exploratory and its provenance requires confirmation.
Technical tests identified a label overflow and a global threshold-control issue.
The current SUVpeak function returns SUVmax.

The test script's paths must be updated for the src/results layout before rerunning it.

## Data
Medical images are not included in this repository.
The exact local data release, access conditions and mask provenance still require verification.
