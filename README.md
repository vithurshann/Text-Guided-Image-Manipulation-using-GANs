# Text-Guided-Image-Manipulation-using-GANs

## Data
TextCaps download link: https://textvqa.org/textcaps/dataset <br>
VizWiz dataset link: https://vizwiz.org/tasks-and-datasets/image-captioning/ <br>
Download textcaps dataset and extract the images to data/textcaps/ <br>
Download vizwiz dataset and extract the images to data/vizwiz/ <br>

## Prerequisites:
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0 $ <br>
Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpu only when installing on a machine without a GPU. <br>

## Training
#### Pre-train DAMSM model
TextCaps : python3 pretrain_DAMSM.py --cfg cfg/DAMSM/textcaps.yml --gpu 0 <br>
VizWiz: python3 pretrain_DAMSM.py --cfg cfg/DAMSM/vizwiz --gpu 0 <br>

#### Main module
TextCaps : python3 main.py --cfg cfg/train_textcaps.yml --gpu 0 <br>
VizWiz: python3 main.py --cfg cfg/train_vizwiz --gpu 0 <br>

#### DCM module
TextCaps : python3 DCM.py --cfg cfg/train_textcaps.yml --gpu 0 <br>
VizWiz: python3 DCM.py --cfg cfg/train_vizwiz --gpu 0 <br>

## Testing
TextCaps : python3 main.py --cfg cfg/eval_textcaps.yml --gpu 0 <br>
VizWiz: python3 main.py --cfg cfg/eval_vizwiz --gpu 0 <br>
To generate images for all captions in the testing dataset, change B_VALIDATION to True in the eval_*.yml. <br>

## Evaluation
For IS: python3 InceptionScore.py <br>
For FID: python3  fid_score.py folder1 folder2 <br>

## Code Structure
code/main.py: the entry point for training the main module and testing ManiGAN. <br>
code/DCM.py: the entry point for training the DCM. <br>
code/trainer.py: creates the main module networks, harnesses and reports the progress of training. <br>
code/trainerDCM.py: creates the DCM networks, harnesses and reports the progress of training. <br>
code/model.py: defines the architecture of ManiGAN. <br>
code/attention.py: defines the spatial and channel-wise attentions. <br>
code/VGGFeatureLoss.py: defines the architecture of the VGG-16. <br>
code/datasets.py: defines the class for loading images and captions. <br>
code/pretrain_DAMSM.py: trains the text and image encoders, harnesses and reports the progress of training. <br>
code/miscc/losses.py: defines and computes the losses for the main module. <br>
code/miscc/lossesDCM.py: defines and computes the losses for DCM. <br>
code/miscc/config.py: creates the option list. <br>
code/miscc/utils.py: additional functions. <br>
model_evaluation/fid_score.py: calculates fid score for folders of images <br>
model_evaluation/inception.py InceptionV3 model for fid score <br>
model_evaluation/InceptionScore.py: calculates inception score for images <br>
model_evaluation/data_loader.py: loads the data for inception score <br>
TextCapsFinalPreProcessing.ipynb:pre-processign results of textcaps datasets <br>
VizWizFinalPreProcessing.ipynb:pre-processign results of Vizwiz datasets <br>
VisualsForReport.ipynb: visuals generated for the report <br>
ModelOutputToResults.ipynb: results of model output processed to text file for visuals <br>

## Acknowledgements
This code borrows heavily from the [ManiGAN](https://github.com/mrlibw/ManiGAN) repository.
