# Estimation of Covid Infection Percentage from CT scans

This repository contains code and weights of the approach developed for the challenge organised by the Workshop on Medical Imaging Analysis for COVID-19 (MIA COVID) associated with the 21st International Conference on Image Analysis and Processing (ICIAP 2021).  
We as [IPLab team](https://iplab.dmi.unict.it/) of the Department of Mathematics and Computer Science of the University of Catania participated in the challenge and entered the final ranking.  

## Description
The challenge requires deep learning algorithms for predicting the percentage of COVID-19 infection in chest CT images of patients. This is a regression task to which we have chosen to participate by adopting state-of-the-art classification algorithms. Specifically, we employ the Inception-v3 neural network together with the data augmentation *mixup* technique. Our results show that the approach is promising despite the limitations given by the constrains defined by the challenge rules.

## Requirements
- python 3.7;
- pytorch 1.7;
- torchvision 0.8.  

The model has been trained on GPU Quadro Quadro RTX 6000.

## Usage
**Case 1**  
Use of pretrained weights to make inference:  
`python run.py --data /directory/of/your/data --action predict --network inceptionv3 --weights best-weights.tar`  

**Case 2**  
Train of the network to replicate our results:  
`python run.py --data /directory/of/your/data --action train --network inceptionv3 --batch 20 --epochs 50 --lr 0.0001 --wd 0.5`  

## Reference
Please cite the following paper if you use the data or pre-trained CNN models.  

@InProceedings{10.1007/978-3-031-13324-4_43,
  author="Napoli Spatafora, Maria Ausilia and Ortis, Alessandro and Battiato, Sebastiano",
  title="Mixup Data Augmentation for COVID-19 Infection Percentage Estimation",
  booktitle="Image Analysis and Processing. ICIAP 2022 Workshops",
  year="2022",
  publisher="Springer International Publishing",
  pages="508--519",
}  

## Acknowledgements and License
The pretrained models can be used under the Creative Common License (Attribution CC BY). Please give appropriate credit, such as providing a link to our paper and to the project web page. 
