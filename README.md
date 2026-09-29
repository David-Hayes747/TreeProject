# MMA3001 Apple Tree Reconstruction Project

## Overview

This repository contains my individual project for MMA3001 – Numerical Methods
and Machine Learning at Monash University.

The project investigates the reconstruction of apple-tree branch structures
using RGB-D imagery collected during summer and winter.

The dataset contains RGB images, corresponding depth images, and ground-truth
branch masks for selected summer observations. The overall aim is to develop
and evaluate a computational method for estimating branch structures that are
partially obscured by foliage.

## Engineering Problem

Accurate identification of tree structure is useful for agricultural robotics,
where knowledge of branch geometry can support perception and path-planning
tasks.

During summer, foliage and fruit obscure much of the underlying branch
structure. Winter observations provide clearer views of the branches.

This project will investigate whether information from RGB-D observations can
be used to reconstruct the underlying branch structure and how accurately the
result agrees with the provided ground truth.

## Inputs

The primary inputs are:

- Summer RGB images
- Summer depth images
- Winter RGB images
- Winter depth images
- Summer ground-truth branch masks

## Outputs

The intended output is a predicted representation of the main trunk and branch
structure of an apple tree.

The predicted structure will be compared with the provided ground-truth masks
using quantitative validation metrics.

## Proposed Method

The computational method is currently under development.

The initial workflow is expected to involve:

1. Loading and pairing RGB, depth and ground-truth images.
2. Pre-processing RGB and depth observations.
3. Extracting visible tree structure.
4. Aligning corresponding seasonal observations where required.
5. Producing a predicted branch representation.
6. Comparing the prediction against the supplied ground truth.
7. Evaluating and improving the method.

A baseline method will be developed before testing more advanced approaches.

## Repository Structure

MMA3001-Project/
│
├── README.md
├── data/
├── src/
├── notebooks/
├── tests/
├── results/
└── docs/

## Installation

Installation instructions will be added as the computational environment and
required dependencies are finalised.

## Usage

Usage instructions will be added as the main computational workflow is
developed.

## Validation

The final method will be evaluated against the provided ground-truth branch
masks.

Potential validation measures include segmentation accuracy and comparison of
predicted and ground-truth branch structures. The final metrics will be selected
and justified during development.

## Limitations

The project is currently under development. Known limitations and unsupported
inputs will be documented as they are identified.

## AI Use

AI tools may be used during the development of this project for tasks such as
explaining concepts, assisting with debugging, reviewing code and discussing
possible computational approaches.

All material AI use will be documented and critically evaluated in accordance
with the MMA3001 project requirements. All final engineering decisions, code,
results and conclusions remain the responsibility of the author.

## Author

David Hayes
MMA3001 – Numerical Methods and Machine Learning
Monash University
2026
