# Rudist FMI Morphometric Analysis

**Author:** Riyad A. Al-Nazzal  
**Institution:** Damascus University – Faculty of Science, Department of Geology  
**Year:** 2026

## Project Overview

This repository provides a C++ computational framework for morphometric analysis of rudist-related features in Formation Microresistivity Imaging (FMI) data from Cretaceous carbonate reservoirs.

The framework integrates digital image processing, geometric feature extraction, Principal Component Analysis (PCA), and Multiple Correspondence Analysis (MCA).

## Methodology

The computational workflow includes:

1. Image enhancement and noise reduction.
2. Morphological segmentation and contour extraction.
3. Measurement of geometric descriptors, including area, elongation, circularity, solidity, and orientation.
4. Multivariate analysis of continuous descriptors using PCA.
5. Categorical morphometric analysis using MCA.
6. Export of annotated images and numerical results.

## Software Requirements

- C++17-compatible compiler
- OpenCV 4.x
- CMake or a compatible C++ build environment

## Source Code

`FMI_Rudist_Morphometric_PCA_MCA.cpp`

## Research Context

Developed in connection with doctoral research on the micropaleontology and carbonate facies of the Cretaceous Massive Formation in the Yousefieh Oilfield, northeastern Syria.

## Citation

Al-Nazzal, R. A. (2026). *Rudist FMI morphometric analysis: Digital image processing, PCA and MCA* [Computer software]. GitHub.

https://github.com/RiyadGeo/Rudist-FMI-Morphometric-Analysis
