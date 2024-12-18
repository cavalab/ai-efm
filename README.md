# AI-EFM

Deep learning models for predicting fetal distress from electronic fetal monitoring.

If you use this repo, please cite: 

Jennifer A. McCoy, Lisa D. Levine, Guangya Wan, Corey Chivers, Joseph Teel, William G. La Cava (2024).
Intrapartum electronic fetal heart rate monitoring to predict acidemia at birth with the use of deep learning. 
*American Journal of Obstetrics and Gynecology (AJOG)*
 [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0002937824005283)  |  [PubMed](https://pubmed.ncbi.nlm.nih.gov/38663662/)  |  [pdf](https://cavalab.org/assets/papers/McCoy%20et%20al.%20-%202024%20-%20Intrapartum%20electronic%20fetal%20heart%20rate%20monitoring.pdf)

# Contributors

- Developed by William La Cava (@lacava), Jennifery McCoy, and Corey Chivers
- contact william.lacava@childrens.harvard.edu





# Install

This project uses tensorflow within a Docker container to train the models. 
For the data preprocessing and result postprocessing, the conda environment is provided in `environment.yml`. 
The conda environment can be installed by running `conda env create` from the repo root directory. 

## Prerequisites

The model training setup needs the following:

- docker
- docker-compose
- Tensorflow Docker: https://www.tensorflow.org/install/docker
- Nvidia docker: https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart

## Setup
1. Clone the repo 
2. Copy and change `mfm.env.template` to `mfm.env` and change the UID to match your user id. You can see your user id by typing 
```
id
```
2. Run
```
docker-compose build
```
To build the docker container. You should only have to do this once. 

3. To start the container, run 
```
docker-compose up -d
```

To start a terminal in the docker container, type `./start_terminal.sh`. 

# Data Preprocessing

Preprocessing scripts for CTU-UHB are in `preprocessing/`. 

## CTU-UHB 

To generate the CTU-UHB training and test sets, first download the database from [physionet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/).
Then navigate to `preprocessing/CTU` and run

```
python preprocess_data.py --data_dir {path-to-ctu-data}
```

# Model Training  

Models are defined in the `methods/` folder and trained using `train_models.py`. 

Type `python train_models.py -h` to see how it is used. 
As an example, the following would train and test a CNN multiscale classifier on the CTU data in `data/CTU`, using a random seed of 42, training on GPU 0, with a classification pH threshold of 7.05, batch size of 128, using data samples preprocessed (see above) to have one hour of tracings with up to 30% missingness and a sample rate of 0.25 Hz and lab order results available within 30 minutes of the end of the trace: 

```
python train_models.py run \
    --datadir data/CTU \
    --ml CNN_multiscale_classification_maxpool \
    --random_state 42 \
    --gpu 0 \
    --savedir results \
    --threshold 7.05 \
    --window 60 \
    --missingness 0.3 \
    --batch_size 128 \
    --sample_rate 0.25 \
    --lab_order_delay 30
```

# Acknowledgments

Financial support: Women’s Reproductive Health Research grant 5 K12 HD 1265-22; T32- HD007440 
Funding sources had no role in study design; in the collection, analysis, and interpretation of data; in the writing of the report; or in the decision to submit the article for publication.
