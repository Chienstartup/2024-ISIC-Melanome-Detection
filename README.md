# ISIC_2024

This repo is only to demonstrate the simplified workflow for the ISIC 2024 competition. Due to the size constraint put by GitHub, I sampled only 5000 rows out of the 400k original dataset.
The reason why I only use the ipynb is that this competition required much of computation power and GPU, so I only use Kaggle notebook and Colab. Here, I breakdown the whole process into 
different notebooks to remind myself the keypoints for each process. 

Based on the EDA for full dataset, I can tell that this dataset is extremely imbalance between target 0 and target 1. While the target 0 data is about 400k rows, 
the target 1 data is only 393 rows. Even though all participants tried to solve the imbalance by using datasets in the previoius competitions, but the resolution of pictures this time is very low
when compared to those in the other competitions. Consequently, if we include images data from previous competition, the performance of our models will drop significantly. Therefore, the majority of
participants actually proceed with only ISIC 2024 imbalance dataset.


### Metric
CV_score metric: partial AUC, maximum 0.2000

```
def custom_metric_binary(y_true, y_pred):
    y_hat = y_pred[:, 1] if y_pred.ndim > 1 else y_pred
    y_true_binary = y_true
    min_tpr = 0.80
    max_fpr = 1 - min_tpr
    v_gt = 1 - y_true_binary
    v_pred = 1 - y_hat
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc
```

### Improvement on Performance:

Only Tabular Data Baseline CV_score: 0.1545

Feature Selection and Hypertuning GBMs CV_score: 0.1688

Combined CNNs CV_score: 0.1886

### Development Process

**Step 1: EDA**
  * Using EDA.ipynb to generate the report.html.
    
**Step 2: Features Engineering**
  * features based on clinical assessments
  * features based on ICC and standardization
  * features used by Kaggle Masters
    
The df_train produced by features_engineering.ipynb is stored in the meta_data folder.

**Step 3: Features Selection**
  * Using lgbm to generate oof_predictions score based on customized metric for 5 folds and stepwise add one feature to find the best possible combination of features.

The df_train_filtered produced by features_selection.ipynb is stored in the meta_data folder.

**Step 4: Ensemble LGB + CAT + XGB and hypertuning for the params**
  * Using optuna to create objectives for 3 models to find params for each.

**Step 5: Training CNN with pretrained Imagenet and df_train_filtered to get out of fold prediction and models**
  * Only change the classifier at the end of pretrained Imagenet
  * Using folds to train 5 fold models
  * Get oof prediction for train data
  * Note: for this step, when Training CNN with pretrained structure notebook, one should upload the sample_5k.hdf to colab by oneself due to GitHub constraint on raw file link.
  * Note: for running this notebook, one should use colab with at least 20 RAM GPU in order to train the CNN with image size of 384.

**Step 6: Combine CNN prediction as new features to the feature_cols and input feature_cols to the ensemble GBMs model**
  * here, I do not predict for the test data. Only for showing improvement on the cv_score
  * 5 oof CNN models due to size limitation cannot be uploaded to GitHub, so I only upload the oof_predictions

With all the steps above, one should get a significant improvement of the pAUC score.
