# Machine Learning Model for Airbnb NYC Listings Price Prediction

This project applies machine learning techniques to predict the price of an Airbnb listing in New York City based on various features such as location, amenities, and host details.

## Steps

### Data Preparation:
- Removed irrelevant and redundant features (host_name, bedrooms, name, description, host_location).
- Handled missing values by imputing numerical columns with the mean and one-hot encoding categorical features.
- Applied winsorization to manage outliers and normalized features using z-score normalization.

### Modeling:
- Started with Linear Regression, followed by more complex models: Decision Tree, Random Forest, and Neural Network.
- Used GridSearch to optimize hyperparameters for improved performance.

### Evaluation:
- Split the data into training and testing sets.
- Evaluated model performance using Mean Squared Error (MSE) and R² score.
- Iterated on models to improve accuracy and generalizability.

## Results and Findings

### Model Performance:
- **Linear Regression**: MSE = 0.587
- **Decision Tree**: MSE = 0.516
- **Random Forest**: MSE = 0.400
- **Updated Random Forest**: MSE = 0.399
- **Neural Network**: MSE ≈ 0.464 (consistent across different layer/unit configurations)

After testing different neural network configurations, the performance plateaued at an MSE of ~0.46, while the Random Forest model achieved a lower MSE of ~0.40. Based on this comparison, Random Forest was chosen as the final model.

### Model Insights:
- The Random Forest model predicts the price closely most of the time.
- The root of MSE indicates that the Random Forest model is off by about 0.63 standard deviations on average, as the data was normalized.
- Given the range of error, the model is best used to generate a price range rather than a single predicted value.

## Tools and Libraries Used:
- Python
- Pandas
- Scikit-learn
- TensorFlow
- NumPy
- Matplotlib / Seaborn
