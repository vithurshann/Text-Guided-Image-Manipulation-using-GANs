# Text-Guided-Image-Manipulation-using-GANs

Data
TextCaps download link: https://textvqa.org/textcaps/dataset
VizWiz dataset link: https://vizwiz.org/tasks-and-datasets/image-captioning/
Download textcaps dataset and extract the images to data/textcaps/
Download vizwiz dataset and extract the images to data/vizwiz/


Training
Pre-train DAMSM model
TextCaps : python3 pretrain_DAMSM.py --cfg cfg/DAMSM/textcaps.yml --gpu 0 
VizWiz: python3 pretrain_DAMSM.py --cfg cfg/DAMSM/vizwiz --gpu 0 

Main module
TextCaps : python3 main.py --cfg cfg/train_textcaps.yml --gpu 0
VizWiz: python3 main.py --cfg cfg/train_vizwiz --gpu 0 

DCM module
TextCaps : python3 DCM.py --cfg cfg/train_textcaps.yml --gpu 0
VizWiz: python3 DCM.py --cfg cfg/train_vizwiz --gpu 0 



Testing
TextCaps : python3 main.py --cfg cfg/eval_textcaps.yml --gpu 0
VizWiz: python3 main.py --cfg cfg/eval_vizwiz --gpu 0 
To generate images for all captions in the testing dataset, change B_VALIDATION to True in the eval_*.yml.



Evaluation
For IS: python3 InceptionScore.py
For FID: python3  fid_score.py folder1 folder2


Code Structure
code/main.py: the entry point for training the main module and testing ManiGAN.
code/DCM.py: the entry point for training the DCM.
code/trainer.py: creates the main module networks, harnesses and reports the progress of training.
code/trainerDCM.py: creates the DCM networks, harnesses and reports the progress of training.
code/model.py: defines the architecture of ManiGAN.
code/attention.py: defines the spatial and channel-wise attentions.
code/VGGFeatureLoss.py: defines the architecture of the VGG-16.
code/datasets.py: defines the class for loading images and captions.
code/pretrain_DAMSM.py: trains the text and image encoders, harnesses and reports the progress of training.
code/miscc/losses.py: defines and computes the losses for the main module.
code/miscc/lossesDCM.py: defines and computes the losses for DCM.
code/miscc/config.py: creates the option list.
code/miscc/utils.py: additional functions.
model_evaluation/fid_score.py: calculates fid score for folders of images
model_evaluation/inception.py InceptionV3 model for fid score
model_evaluation/InceptionScore.py: calculates inception score for images
model_evaluation/data_loader.py: loads the data for inception score
TextCapsFinalPreProcessing.ipynb:pre-processign results of textcaps datasets 
VizWizFinalPreProcessing.ipynb:pre-processign results of Vizwiz datasets 
VisualsForReport.ipynb: visuals generated for the report
ModelOutputToResults.ipynb: results of model output processed to text file for visuals 



For pyTorch installation:
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0 $
Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpu only when installing on a machine without a GPU.

# Acknowledgements
This code borrows heavily from the [ManiGAN](https://github.com/mrlibw/ManiGAN) repository.
