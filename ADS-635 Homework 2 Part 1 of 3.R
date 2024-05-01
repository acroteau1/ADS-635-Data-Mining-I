#######################################################
# Homework 2 Question 1
# ADS-635: Data Mining I
# Alison Croteau
# Created: 9/28/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(pls)
library(ISLR)
library(caret)
library(tidyverse)
library(glmnet)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import data
college <- read.csv("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\College.csv", 
                    header = T)

#######################################################
# Variable Information
# Private : Public/private indicator
# Apps : Number of applications received
# Accept : Number of applicants accepted
# Enroll : Number of new students enrolled
# Top10perc : New students from top 10% of high school class
# Top25perc : New students from top 25% of high school class
# F.Undergrad : Number of full-time undergraduates
# P.Undergrad : Number of part-time undergraduates
# Outstate : Out-of-state tuition
# Room.Board : Room and board costs
# Books : Estimated book costs
# Personal : Estimated personal spending
# PhD : Percent of faculty with Ph.D.’s
# Terminal : Percent of faculty with terminal degree
# S.F.Ratio : Student/faculty ratio
# perc.alumni : Percent of alumni who donate
# Expend : Instructional expenditure per student
# Grad.Rate : Graduation rate
#######################################################

# Remove the unnamed college name column
college <- college[, -1]

# Consider benefits of changing 'Private' variable from chr to binary?
# Of note: All numeric data except for S.F.Ratio is integer over numeric

# 1a - Split the data set into a training set and a test set
# Set a seed for reproducibility, create row indices vector for training set, 
# then create test and training data by selecting rows in or not in the 
# indices vector.
set.seed(1)
college_indices <- createDataPartition(college$Apps, p = 0.75, list = FALSE)
college_train <- college[college_indices,]
college_test <- college[-college_indices,]

# Center and scale the variables to standardize features
preObj <- preProcess(college_train, method = c('center', 'scale'))
college_train <- predict(preObj, college_train)
college_test <- predict(preObj, college_test)

# Extract the target variable
y_train <- college_train$Apps
y_test <- college_test$Apps

# Create dummy variables
dum <- dummyVars(Apps ~ ., data = college_train)
x_train <- predict(dum, college_train)
x_test <- predict(dum, college_test)

# 1a - Fit a linear model using least squares on the training set, and 
# report the test error obtained.
model <- lm(Apps ~ ., data = college_train)
pred <- predict(model, college_test)
model_info <- postResample(pred, college_test$Apps)
model_info

summary(college$Apps)

# Diagnosing error "factor X has new levels"
factor_columns <- names(college_train)[sapply(college_train, is.factor)]
for (col in factor_columns) {
  training_levels <- levels(college_train[[col]])
  test_levels <- levels(college_test[[col]])
  new_levels <- setdiff(test_levels, training_levels)
  if (length(new_levels) > 0) {
    print(paste("Column", col, "has new levels:", toString(new_levels)))
  }
}

# Note: error was caused by the unnamed column containing the college name.
# Error resolved by adding snippet removing column from analysis as a temporary
# measure at line 45.

# 1b - Fit a ridge regression model on the training set, with λ chosen by 
# cross-validation. Report the test error obtained.
# Convert data to matrix format which is required by glmnet
x_train_matrix <- as.matrix(x_train)
x_test_matrix <- as.matrix(x_test)

# Fit ridge regression with CV
set.seed(1)
cv.out <- cv.glmnet(x_train_matrix, y_train, alpha = 0)
plot(cv.out)

# Predict using the best lambda
ridge_pred <- predict(cv.out, s = cv.out$lambda.min, newx = x_test_matrix)

# Report test error
rmse_test <- sqrt(mean((ridge_pred - y_test)^2))
rmse_test

# 1c - Fit a lasso model on the training set, with λ chosen by cross-
# validation. Report the test error obtained, along with the number of 
# non-zero coefficient estimates.
cv.lasso <- glmnet(x_train_matrix, y_train, alpha = 1)
plot(cv.lasso)

# Predict using the best lambda
lasso_pred <- predict(cv.lasso, s = cv.lasso$lambda.min, newx = x_test_matrix)

# Compute RMSE on test
rmse_test_lasso <- sqrt(mean((lasso_pred - y_test)^2))
rmse_test_lasso

# Extract coefficients and count of non-zero coefficients
lasso_coefs <- coef(cv.lasso, s = cv.lasso$lambda.min)
nz_coefs <- sum(lasso_coefs != 0)
nz_coefs

# 1d - Fit a PCR model on the training set. Report the test error 
# obtained, along with justification for the choice of “k”.
# Perform PCR on the data and evaluate its test set performance.
pcr_model <- pcr(Apps ~ ., data = college, scale = TRUE,
               validation = "CV")
summary(pcr_model)

# Extract validation results and plot
validationplot(pcr_model, val.type = "MSEP")

# Perform PCR on the training data and evaluate its test set performance.
set.seed(1)
pcr.fit <- pcr(Apps ~ ., data = college_train, scale = TRUE, 
               validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")

# Compute the test MSE
pcr.pred <- predict(pcr.fit, newdata=college_test, ncomp = 16)
mean((pcr.pred - y_test)^2)

# 1e - Fit a PLS model on the training set. Report the test error 
# obtained, along with justification for the choice of “k”.
set.seed(1)
pls.fit <- plsr(Apps ~ ., data = college_train, scale = TRUE, 
                validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")

# Evaluate the corresponding test set MSE for M = 6.
pls.pred <- predict(pls.fit, newdata=college_test, ncomp = 6)
mean((pls.pred - y_test)^2)