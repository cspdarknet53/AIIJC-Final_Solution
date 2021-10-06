# Classification of traffic signs

Russian version: [README.md](docs/README.md)

## Installation
1. Clone this repository:
   ```shell
   git clone https://github.com/2gis/signs_classification_aij2021.git 
   cd signs_classification_aij2021
   ```
2. Install required python packages:
   ```shell
    pip install -r requirements.txt
   ```
3. Download the data. By default, it is assumed that the dataset is located
   in the `data/` folder in the same project. If this is not the case, then change
   `DEFAULT_DATA_PATH` from `pipeline/constants.py` on the correct path.
4. Download weights from https://drive.google.com/file/d/1rPQHBvp8w_F9Nrtbr1p6lAx__zCTMXAu/view?usp=sharing and put them into ./Web app/app .
   
## Neural network training

By default, experiments are saved in the `experiments/` folder. If you want to
do it differently, you need to change `DEFAULT_EXPERIMENTS_SAVE_PATH` from
`pipeline/constants.py` on the correct path. The weights of the best model are stored
as `experiments/experiment_name/best.pth`.
### Input parameters
- **exp_name** - the name of the experiment (this folder will be created in the
  `experiments/` folder)
- **n_epochs** - number of epochs for training
- **model_name** - name of the network encoder. The available encoders are described in the
  `ENCODERS` dictionary from `pipeline/models.py`
- **batch_size** - batch size
- **device** - device on which the calculations will be performed

### Run

   ```shell
   python -m pipeline.train \
       --exp_name baseline \
       --n_epochs 50 \
       --model_name resnet18 \
       --batch_size 8 \
       --device cuda:0
   ```

## Test script

The script calculates the answers for the test images and saves the result to the
experiment folder in a file named `submit.csv`.
### Input parameters
- **exp_name** - name of the experiment from which the model is being tested
- **model_name** - name of the encoder of the model to be tested
- **batch_size** - batch size
- **device** - device on which the calculations will be performed

### Run
   ```shell
   python -m pipeline.generate_submission \
       --exp_name baseline \
       --model_name resnet18 \
       --batch_size 8 \
       --device cuda:0
   ```
