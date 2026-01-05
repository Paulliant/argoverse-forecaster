# Argoverse Forecaster

This project is developed as part of the Python Programming course for the 2023 semester. It addresses the problem of trajectory prediction by proposing a **feature extraction-based approach**. By optimizing traditional regression models and incorporating neural network features, this project demonstrates the superiority of algorithms that rely on meaningful, manually extracted features.

Team Name: LocationMind

Team Member: 19210121-陈铭豪

## Reference

Please check the details in
[**motion_prediction_feature_extraction.pdf**](motion_prediction_feature_extraction.pdf)

## Methodology

### Iterative Regression Prediction

Please refer to [**code.ipynb**](code.ipynb) for implementation details.

### VectorNet Feature Embedding

The VectorNet module utilizes a reimplementation available [**here**](https://github.com/Liang-ZX/VectorNet). To prepare for using VectorNet:

1. Install the Argoverse API by following the instructions at [**Argoverse GitHub repository**](https://github.com/argoverse/argoverse-api).
2. Ensure that the HD map data is placed in the specified location as per the Argoverse API documentation.

**Setup and Training:**

Navigate to the VectorNet directory:
```bash
cd /path/to/VectorNet
```

Start the training process:
```bash
python train.py
```

The training losses will be output to both the terminal and [**log.txt**](log.txt).

## Summary

The project involves using advanced techniques to enhance trajectory prediction, showcasing the importance of both classical and deep learning-based feature extraction methods.
