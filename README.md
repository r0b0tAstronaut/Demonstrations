# Nicholas Califano
### Example ML Projects

I've written some examples projects highlighting some use cases of different machine learning methods. These are most toy examples on relatively small datasets to showcase some examples. The performance of these methods is often acceptable, but not spectacular due to intentionally using a relatively small number of training parameters.

### Examples
| Example          | Dataset                                     | Demonstration of:                                   | Results                        | Potential Future Work               |
|------------------|---------------------------------------------|-----------------------------------------------------|--------------------------------|--------------------------------|
| cifar.py         | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | Libraries - Tensorflow <br /> Models - Convolutional Neural Networks, ResNet18 <br /> Techniques - Regularization, Data Augmentation, Training Validation, Pretrained Model | Up to ~86% Accuracy, 10 Categories, on Test Data | Tweak models without increasing complexity, tweak training hyperparameters and training schedules without greatly increasing training time. |
| house_prices.py      | [Housing Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) | Libraries - Sklearn, Pandas <br /> Model - Gradient Boosted Decision Trees <br /> Techniques - Bagging, Hyperparameter Search, Crossfold Validation <br /> Other - Data Cleaning | ~0.89 R2 on Validation Data | Feature Engineering and Better Data Cleaning | 
