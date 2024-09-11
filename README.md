# ISIC_2024

This repo is only to demonstrate the simplified workflow for the ISIC 2024 competition. Due to the size constraint put by GitHub, I sampled only 5000 rows out of the 400k original dataset.
The reason why I only use the ipynb is that this competition required much of computation power and GPU, so I only use Kaggle notebook and Colab. Here, I breakdown the whole process into 
different notebooks to remind myself the keypoints for each process. 

Based on the EDA for full dataset, I can tell that this dataset is extremely imbalance between target 0 and target 1. While the target 0 data is about 400k rows, 
the target 1 data is only 393 rows. Even though all participants tried to solve the imbalance by using datasets in the previoius competitions, but the resolution of pictures this time is very low
when compared to those in the other competitions. Consequently, if we include images data from previous competition, the performance of our models will drop significantly. Therefore, the majority of
participants actually proceed with only ISIC 2024 imbalance dataset.

#Step 1: EDA
  * Using EDA.ipynb to generate the report.html.
    
#Step 2: Features Engineering
  * features based on clinical assessments
  * features based on ICC and standardization
  * features used by Kaggle Masters
    
The df_train produced by features_engineering.ipynb is stored in the meta_data folder.

#Step 3: Features Selection
  * Using lgbm to generate oof_predictions score based on customized metric for 5 folds and stepwise add one feature to find the best possible combination of features.

The df_train_filtered produced by features_selection.ipynb is stored in the meta_data folder.

#Step 4: Ensemble LGB + CAT + XGB and hypertuning for the params
  * Using optuna to create objectives for 3 models to find params for each.

#Step 5: Training CNN with pretrained Imagenet and df_train_filtered to get out of fold prediction and models
  * Only change the classifier at the end of pretrained Imagenet
  * Using folds to train 5 fold models
  * Get oof prediction for train data
