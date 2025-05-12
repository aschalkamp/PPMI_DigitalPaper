# Digital risk score and pathological markers in Parkinson's disease

This is the code repository accompanying the manuscript **Digital risk score sensitively identifies presence of α-synuclein aggregation or dopaminergic deficit** (https://www.medrxiv.org/content/10.1101/2024.09.05.24313156v1).

## Summary

Using data from PPMI we derive a digital risk score and compare it to established biological and pathological markers for Parkinson's disease. The digital risk score is based on 14 timeseries features derived from data collected with the Verily Study Watch encompassing sleep, physical activity, and vital signs. A risk score is trained distinguishing diagnosed cases from healthy controls. A separate group formed of individuals with prodromal markers and risk factors for Parkinson's disease is used to compare the digital risk score to the known MDS prodromal score and dopaminergic imaging and alpha-synuclein pathology.

## Structure of this repository

This repository is split into

    1_DataPrepocessing: which deals with loading of data from PPMI via an adapted version of pympi, cleaning of data, and extraction of progression estimates
    2_Analysis: all conducted relevant analysis (residuals, classification model, correlations, group comparisons)
    scripts: helper functions
    environment: anaconda .yml file to recreate python environment. 
    Relies on the pypmi package (an adapted version: https://github.com/aschalkamp/pypmi/tree/master)
