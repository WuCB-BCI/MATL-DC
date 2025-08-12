# MATL-DC: A Multi-domain Aggregation Transfer Learning Framework for EEG Emotion Recognition with Domain-Class Prototype under Unseen Targets 
*   A Pytorch implementation of our under reviewed paper "MATL-DC: A Multi-domain Aggregation Transfer Learning Framework for EEG Emotion Recognition with Domain-Class Prototype under Unseen Targets".
# Installation
*   Python 3.8
*   Pytorch 2.0.0
*   NVIDIA CUDA 11.8
*   NVIDIA CUDNN 8700
*   Numpy 1.24.3
*   Scikit-learn 0.22.1
*   scipy 1.5.2 
*   GPU NVIDA GeForce RTX 3090
# Databases
*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html ""), [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html ""), [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html "") 
# Training
*   Data Process Module: utils.py
*   One-Hot processing: DataProcessing_OneHot.py
*   Dynamically Updating Gradients: StepwiseLR_GRL.py
*   MATL-DC Framework : MATL-DC framework.py
*   Pairwise Learning Module: Pairwise_Learning_Modual.py 
*   MATL_DC_Train_verification: MATL_DC_Train.py

# Usage
*   After modify setting (path, etc), just run the main function in the MATL_DC_Train.py.
