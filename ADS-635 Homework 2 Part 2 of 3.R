#######################################################
# Homework 2 Question 2
# ADS-635: Data Mining I
# Alison Croteau
# Created: 9/28/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(dplyr)
library(ggplot2)
library(ISLR2)
library(glmnet)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import data
data(Caravan)
head(Caravan)

#######################################################
# Variable Information
# Each training record consists of 86 attributes, containing sociodemographic data 
# (attribute 1-43) and product ownership (attributes 44-86).The sociodemographic 
# data is derived from zip codes. All customers living in areas with the same zip 
# code have the same sociodemographic attributes.
# Variable 86 contains caravan insurance policy binary.
# Can we predict on test who will want caravan insurance?
#######################################################

# 2a - Compute the OLS estimates.
# Convert the factor variable to numeric
Caravan$PurchaseNumeric <- as.numeric(Caravan$Purchase) - 1

# Use OLS to fit a linear regression model
model <- lm(PurchaseNumeric ~ . - Purchase, data = Caravan)
summary(model)

# Use OLS to predict likelihood of caravan policy purchase
predicted_values <- predict(model, newdata = Caravan)

# Determine threshold for result to be counted as a purchase based
# on binary factor variable
predicted_class <- ifelse(predicted_values > 0.5, "Yes", "No")
table(Actual = Caravan$Purchase, Predicted = predicted_class)

((5473 + 2) / (5473 + 1 + 346 + 2))

# 2b - Compare the OLS estimates with those obtained from Forwards Selection.
model_full <- lm(PurchaseNumeric ~ . - Purchase, data = Caravan)
model_null <- lm(PurchaseNumeric ~ 1, data = Caravan)
model_fwd <- step(model_null, direction = "forward", 
                  scope = list(lower = model_null, upper=model_full), trace=1)
summary(model_full)$coefficients
summary(model_fwd)$coefficients

# Use OLS to predict likelihood of caravan policy purchase
predicted_values_fwd <- predict(model_fwd, newdata = Caravan)

# Determine threshold for result to be counted as a purchase based
# on binary factor variable
predicted_class_fwd <- ifelse(predicted_values_fwd > 0.5, "Yes", "No")
table(Actual = Caravan$Purchase, Predicted = predicted_class_fwd)

((5473 + 1) / (5473 + 1 + 347 + 1))

# 2c - Compare the OLS estimates with those obtained from Backwards Selection. 
model_bwd <- step(model_full, direction="backward")
summary(model_full)$coefficients
summary(model_bwd)$coefficients

# Use OLS to predict likelihood of caravan policy purchase
predicted_values_bwd <- predict(model_bwd, newdata = Caravan)

# Determine threshold for result to be counted as a purchase based
# on binary factor variable
predicted_class_bwd <- ifelse(predicted_values_bwd > 0.5, "Yes", "No")
table(Actual = Caravan$Purchase, Predicted = predicted_class_bwd)

((5473 + 1) / (5473 + 1 + 347 + 1))

# 2d - Compare the OLS estimates with those obtained from Lasso regression.
X <- as.matrix(Caravan[, -which(names(Caravan) %in% c("Purchase", "PurchaseNumeric"))])
Y <- Caravan$PurchaseNumeric

set.seed(123)
lasso <- cv.glmnet(X, Y, alpha = 1)
best_lambda <- lasso$lambda.min
coef_lasso <- coef(lasso, s=best_lambda)
coef_lasso

new_data <- X
predicted_values_lasso <- predict(lasso, newx = new_data, 
                                  s=best_lambda, type="response")
predicted_class_lasso <- ifelse(predicted_values_lasso > 0.5, "Yes", "No")
table(Actual = Caravan$Purchase, Predicted = predicted_class_lasso)

((5473 + 1) / (5473 + 1 + 347 + 1))

# 2e - Compare the OLS estimates with those obtained from Ridge regression.
X <- as.matrix(Caravan[, -which(names(Caravan) %in% c("Purchase", "PurchaseNumeric"))])
Y <- Caravan$PurchaseNumeric

ridge <- cv.glmnet(X, Y, alpha = 0)
best_lambda_ridge <- ridge$lambda.min
coef_ridge <- coef(ridge, s=best_lambda_ridge)
coef_ridge

p_ridge <- predict(ridge, newx = X, 
                   s = best_lambda_ridge, type = "response")[,1]
p_class_ridge <- ifelse(p_ridge > 0.5, "Yes", "No")
table(Actual = Caravan$Purchase, Predicted = p_class_ridge)

((5474 + 1) / (5474 + 0 + 347 + 1))

