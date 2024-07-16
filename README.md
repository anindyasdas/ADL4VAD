# Adaptive Deviation Learning
Official Pytorch Implementation of Adaptive Deviation Learning (ADL)
## Installation
Our results were computed using Python 3.8, with packages and respective version noted in
`requirements.txt`.
Please install the dependencies 
```
pip install -r requirements.txt
```
## Datasets
- To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad) Extract and move it to ./data/ folder.
- To train on the Visual Anomaly Dataset (VisA) [download](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) Extract and move it to ./data/ folder
- The [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source. Extract and move it to ./data/ folder.

## Training
Pass the folder containing the training dataset to the **adl_train.py** script as the --dataset_root argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the  epochs (--epochs), path to store checkpoint path (--experiment_dir), checkpoint weight name (--weight_name), contamination ratio (--cont).
Example:

```
python adl_train.py --dataset_root=./data/mvtec_anomaly_detection --dataset=mvtec --anomaly_source_path ./data/dtd/images --classname=bottle --experiment_dir=./experiment --epochs=25 --cont=0.1 --weight_name=model_bal_10.pkl --report_name=result_report_mvtec_bal10
```

## Evaluation
For Evaluation run **eval_adl.py** script with required arguments.
Example:
```
python eval_adl.py --dataset_root=./data/mvtec_anomaly_detection --dataset=mvtec --classname=bottle --experiment_dir=./experiment --cont=0.1  --weight_name=model_bal_10.pkl --report_name=result_report_mvtec_bal10
```
