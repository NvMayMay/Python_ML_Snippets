'''Backward elimination feature selection'''
# Define a significance level
significance_level = 0.05
while True:
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Get the highest p-value in the model
    max_p_value = model.pvalues.max()
    
    # Check if the highest p-value is greater than the significance level
    if max_p_value > significance_level:
        # Identify the feature with the highest p-value
        feature_to_remove = model.pvalues.idxmax()
        print(f"Removing feature: {feature_to_remove} with p-value: {max_p_value}")
        
        # Drop the feature
        X = X.drop(columns=[feature_to_remove])
    else:
        break
# Display the final model summary
print(model.summary())

'''Forward selection feature selection'''
def forward_selection(X, y):
    remaining_features = set(X.columns)
    selected_features = []
    current_score = 0.0
    best_score = 0.0
    
    while remaining_features:
        scores_with_candidates = []
        
        # Loop through remaining features
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions and calculate R-squared
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            # Record the score with the current feature
            scores_with_candidates.append((score, feature))
        
        # Sort candidates by score (highest score first)
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]
        
        # If adding the feature improves the score, add it to the model
        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break
    return selected_features
# Run forward selection
best_features = forward_selection(X, y)
print("Selected features using Forward Selection:", best_features)

'''LASSO feature selection'''
# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)
# Train the model on the training data
lasso_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = lasso_model.predict(X_test)
# Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')
# Try different alpha values and compare the results
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'Alpha: {alpha}, R-squared score: {r2}, Coefficients: {lasso_model.coef_}')