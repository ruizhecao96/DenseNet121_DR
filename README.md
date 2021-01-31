# DenseNet121_DR
Using transfer learning to classify diabetic retinopathy fundus images

# Dataset
```https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid```

3. Dataset dir:
```
IDRID_Data/
    -train_image/
    -test_image/
    -train.csv
    -test.csv
```
4. Prepare:\
Change data_EDA directory. Run data_EDA to convert label, create TFRecord files, and show label distribution
```
python input_pipeline/data_EDA.py
```
5. Training and fine tuning:\
Change dataset directory in config.gin, set train flag to Ture in main.py
```
python main.py
```
6. Evaluation:\
Set train flag to False, change model directory to the saved model in log folder in main.py
```
python main.py
```
7. Hyperparameter tuning:\
Change config.gin path to your own path
```
python tune.py
```
8. Deep Visualization:\
Change model_dir and dataset_dir to your own directory
```
python grad_cam.py
```
