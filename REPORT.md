# BigMart Sales Prediction: Project Report

## Summary

This project predicts how much of each product (`Item_Outlet_Sales`) a BigMart outlet sells, using product and store details. The final model is a tuned HistGradientBoosting regressor. On held-out validation data it reaches **RMSE 1024.34** and **R² 0.614**, meaning it explains about 61% of the variation in sales. It was used to produce predictions for all 5,681 test rows.

## Data

| File | Rows | Columns | Notes |
| --- | --- | --- | --- |
| `Train.csv` | 8,523 | 12 | Includes the target, `Item_Outlet_Sales` |
| `Test.csv` | 5,681 | 11 | No target; used only for the final predictions |

The data covers 1,559 products, 16 product types and 10 outlets of 4 types, with outlets established between 1985 and 2009. Sales range from 33 to 13,087, with a mean of 2,181 and a median of 1,794, so the distribution is skewed to the right.

**Data quality issues found:**

- `Outlet_Size` is missing in 2,410 rows (28%), and `Item_Weight` in 1,463 rows (17%).
- `Item_Visibility` is 0 in 526 training rows and 353 test rows. Every product sold in a store must be visible, so a 0 is treated as a missing value.
- `Item_Fat_Content` uses inconsistent labels for the same thing: "Low Fat", "LF" and "low fat", and "Regular" and "reg".

**Notable patterns:**

- Store type matters most. Average sales are 340 in grocery stores and 3,694 in Supermarket Type3 outlets.
- Price is the strongest single numeric signal: `Item_MRP` has a correlation of 0.57 with sales.

## Approach

The work is split into two notebooks, run in this order:

1. **`bigmart_sales_data.ipynb`**: cleaning and feature engineering
2. **`bigmart_modeling_and_prediction.ipynb`**: model comparison, tuning and final predictions

### Cleaning

- Unified the fat-content labels into "Low Fat" and "Regular". Non-consumable products (household, health and hygiene, etc.) are labeled "Non-Edible".
- Filled missing `Item_Weight` with the same product's average weight, falling back to the average for its product type.
- Filled missing `Outlet_Size` with the most common size for that outlet type.
- Replaced zero visibilities with the product's average visibility over its non-zero rows.

All fill values are learned from the training data only and then applied to the validation and test data. Statistics from those sets never influence their own cleaning.

### New features

| Feature | Meaning |
| --- | --- |
| `Item_Category` | Food, Drinks or Non-Consumable, taken from the first two letters of the product ID |
| `Outlet_Age` | Years the outlet had been open in 2013, the year the data was collected |
| `Item_Visibility_Mean` | The product's average visibility in other rows |
| `Visibility_Ratio` | How visible the product is in this store compared with its average |

### Modeling

- The training data was split 80/20 into training (6,818 rows) and validation (1,705 rows) parts. The split happens before cleaning, so the validation scores are honest.
- Numeric features are standardized and categorical features are one-hot encoded, giving 48 model inputs. Preprocessing and model are combined in one scikit-learn pipeline.
- Four models were compared, then the HistGradientBoosting model was tuned with a 3-fold cross-validated grid search (48 parameter combinations).
- The tuned model was retrained on all 8,523 training rows before predicting the test set.

## Results

Validation scores (lower RMSE and higher R² are better):

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| **HistGradientBoosting (tuned, final)** | **715.94** | **1024.34** | **0.6139** |
| SVR (scaled target) | 719.49 | 1035.03 | 0.6058 |
| HistGradientBoosting (default) | 729.97 | 1051.22 | 0.5934 |
| Random Forest | 744.10 | 1069.02 | 0.5795 |
| Linear Regression | 791.59 | 1069.15 | 0.5794 |

Best parameters: `learning_rate=0.05`, `max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=0.0`. These favor a small, strongly regularized model, which suggests the data has limited signal and overfitting is the main risk.

All models end up close together (R² between 0.58 and 0.61). The available features explain only about 60% of sales; the rest depends on factors not in the data, such as promotions, seasonality and local demand.

## Final predictions

The output is `bigmart_final_predictions.csv`, with one row per test item and outlet (`Item_Identifier`, `Outlet_Identifier`, `Item_Outlet_Sales`).

| | Predicted sales |
| --- | --- |
| Rows | 5,681 |
| Mean | 2,180 |
| Median | 2,091 |
| Range | 204 to 6,489 |

The predicted mean (2,180) matches the training mean (2,181). The predicted range is narrower than the actual one (33 to 13,087), because the model is cautious at the extremes. On the validation set, the 10% best-selling rows average 5,636 in actual sales but only 4,030 in predictions. The 10% weakest rows average 181 actual but 446 predicted.

## Limitations and next steps

- About 40% of the variation in sales is unexplained. More data about the stores or time periods would help more than further model tuning.
- Try predicting the log of sales: it may handle the skewed target and the high-sales cases better.
- Tune SVR, which came close to the best model with default settings.
- Inside the grid search, fill values were computed over the whole training part, so the cross-validation scores may be very slightly optimistic. Moving the cleaning into the scikit-learn pipeline would remove this.

## How to run

Requirements: Python with `pandas`, `scikit-learn`, `matplotlib`, `seaborn` and Jupyter.

1. Put `Train.csv` and `Test.csv` in the project folder.
2. Run `bigmart_sales_data.ipynb`. It writes `prepared_train.csv`, `prepared_test.csv`, `prepared_train_part.csv` and `prepared_valid_part.csv`.
3. Run `bigmart_modeling_and_prediction.ipynb`. It writes `bigmart_final_predictions.csv`.
