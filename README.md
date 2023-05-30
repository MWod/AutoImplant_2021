# AutoImplant 2021 - AGH Team Contribution

# Work In Progress

## Introduction

The respository contains source code developed by AGH Team for the [AutoImplant Challenge](https://autoimplant2021.grand-challenge.org/Home/) organized jointly with the [MICCAI 2021](https://miccai2021.org/en/) conference, and then further extended.

Our contribution scored the 1st place in all the challenge tasks.

We release the complete source code to support the open and reproducible research.

## Methods

The method aims to propose a method for fully automatic cranial implant reconstruction.
The input to the method is a binary mask of the skull with a cranial defect.
The output of the method is an automatically calculated reconstruction of the cranial defect, or optionally the implant itself (in .nrrd format or .stl ready for 3-D printing). 

![Pipeline](https://github.com/MWod/AutoImplant_2021/blob/main/GitHub_Pipeline.png)

![3D](https://github.com/MWod/AutoImplant_2021/blob/main/Github_3D.png)

![MR](https://github.com/MWod/AutoImplant_2021/blob/main/GitHub_AR.png)

The full descrption of the proposed method is presented in [Link](https://arxiv.org/abs/2204.06310)

Please find a link to a movie presenting the method outcome: [The Movie](https://www.youtube.com/watch?v=a1IMMtt3ovc)

## Code structure

The code structure is follows:

* The main files:
    * [pipeline.py](pipeline.py) - The main file implementing the method pipeline. This is the file you want to use to perform the defect reconstruction or the implant modeling.
    * [example.py](example.py) - An example how to use the pipeline.py file.

* The utility files:
    * [paths.py](paths.py) - The file contains the absolute and relative paths used within the training/evaluation scripts. You do not need to change the paths for using the source code. However, if you plan to reproduce the results you should replace the absolute paths at the beginning of the file.
    * [utils.py](utils.py) - The file contains numerous simple utility functions used in all the remaining files.

* The data pre/post-processing files:
    * [csv_creator.py](csv_creator.py) - This file creates CSV files from the original dataset structure.
    * [datasets.py](datasets.py) - This file implements the PyTorch dataset and dataloader for the AutoImplant datasets (assuming CSV files created from the [*csv_creator.py*](csv_creator.py))
    * [preprocessing.py](preprocessing.py) - This file performs the initial preprocessing (boundary cropping, resampling, padding to the same shape).
    * [skull_refinement_training_set_creation.py](skull_refinement_training_set_creation.py) - This file creates the training set for the defect refinement.

* The augmentation files:
    * [augmentation.py](augmentation.py) - This file contains basic geometrical augmentation (affine transformation / flipping).
    * [image_registration.py](image_registration.py) - This file performs the cross-case image registration to perform the offline augmentation. This augmentation is the most influential in terms of improving the method generalizability.

* The training files:
    * [training_implant_reconstruction.py](training_implant_reconstruction.py) - This file is used to train the defect reconstruction / defect refinement (based on PyTorch Lightning).
    * [training_variational_augmentation.py](training_variational_augmentation.py) - This file is used to train the variational augmentation (based on PyTorch Lightning).
    
* The evaluation files:
    * [evaluation.py](evaluation.py) - This file performs the evaluation of a given model (for the defect reconstruction / defect refinement.)
    * [evaluation_metrics.py](evaluation_metrics.py) - This file contains the evaluation metrics. NOTE: File from the AutoImplant 2021 organizers.
    * [run_evaluation.py](run_evaluation.py) - This file runs the evaluation on the full resolution for the whole processing pipeline.
    

## How to perform the defect reconstruction

Please find a detailed example in the [example.py](example.py) file.

## How to reproduce the results

1) Change and create absolute paths in [paths.py](paths.py) 
2) Use [csv_creator.py](csv_creator.py) to create the training/validation/testing CSV files.
3) Use [preprocessing.py](preprocessing.py) to preprocess the training/validation files.
4) (Optional) Use the [image_registration.py](image_registration.py) to perform the cross-case image registration to augment the dataset. NOTE: This requires separate library and may take a lot of time (performs more than 20 000 3-D registrations). For an access to the augmented training set, please see [Pretrained models](#pretrained-models)
5) (Optional) Use the [vae_augmented_training_set_creation.py](vae_augmented_training_set_creation.py) to perform the image generation-based data augmentation.
6) Use the [training_implant_reconstruction.py](training_implant_reconstruction.py) to train the defect reconstruction model.
7) (Optional) Use the [skull_refinement_training_set_creation.py](skull_refinement_training_set_creation.py) to create the training set for the defect refinement.
8) (Optional) Use the [training_implant_reconstruction.py](training_implant_reconstruction.py) again, this time for the defect refinement.
9) Use the [pipeline.py](pipeline.py) and/or [example.py](example.py) to perform the defect reconstruction / implant modeling.

Exemplary experiments/evaluation files are in the [Experiments](experiments/) and [Evaluation](evaluation/) respectively.

## Pretrained models

If you just want to use the model, without reproducing the method by yourself, please contact <wodzinski@agh.edu.pl> for the access to the pretrained models.

## Future work

In the future work we plan to perform the implant modeling directly, without the defect reconstruction, using deep reinforcment learning. Please stay tuned if you are interested in this topic.

## References

If you found the source code useful please cite:
* M. Wodzinski, M. Daniol, M. Socha, D. Hemmerling, M. Stanuch, A. Skalski, *Deep Learning-based Framework for Automatic Cranial Defect Reconstruction and Implant Modeling*, Computer Methods and Programs in Biomedicine, 2022, DOI: 10.1016/j.cmpb.2022.107173 [Link](https://www.sciencedirect.com/science/article/pii/S0169260722005545)
* M. Wodzinski, M. Daniol, D. Hemmerling, *Improving the Automatic Cranial Implant Design in Cranioplasty by Linking Different Datasets*,  AutoImplant 2021: Towards the Automization of Cranial Implant Design in Craniplasty II, DOI: 10.1007/978-3-030-92652-6_4 [Link](https://link.springer.com/chapter/10.1007/978-3-030-92652-6_4)
* J. Li, et. al., *Towards Clinical Applicability and Computation Efficiency in Automatic Cranial Implant Design: An overview of the AutoImplant 2021 Cranial Implant Design Challenge*, In Review [Link](TODO)

## Acknowledgments

We would like to thank:
* The AutoImplant challenge organziers for evaluating our ablation studies on the testing set - [Link](https://autoimplant2021.grand-challenge.org/Organizers/).
* The MedApp S.A for giving us the access to the CarnaLife Holo technology - [Link](https://medapp.pl/en/).
