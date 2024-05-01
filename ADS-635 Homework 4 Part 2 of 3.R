#######################################################
# Homework 4 Question 2
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/24/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(rpart) # trees
library(gbm) # boosting
library(randomForest) # RF / bagging
library(ISLR)
library(class) # kNN
library(caret) # confusion matrix
# install.packages("pdp")
library(pdp) # partial dependence plots

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import the downloaded RData
head(pima)

#######################################################
# Variable Information
# 532 observations of 9 variables
# npregnant - int - Number of times pregnant
#
# glucose - int - Plasma glucose concentration at 2 hours 
# in an oral glucose tolerance test
#
# diastolic.bp - int - Diastolic blood pressure (mm Hg)
#
# skinfold.thickness - int - Triceps skin fold thickness (mm)
#
# bmi - num - Body mass index (weight in kg/(height in metres squared))
#
# pedigree - num - Diabetes pedigree function
#
# age - Age (years)
#
# classdigit - factor - test whether the patient shows 
# signs of diabetes (coded 0 if negative, 1 if positive)
#
# class - factor - test whether the patient shows
# signs of diabetes (coded diabetic and normal)
#######################################################

# Split into test and training
set.seed(123)
train_indices <- sample(1:nrow(pima), 0.7 * nrow(pima))
train_data <- pima[train_indices, ]
test_data <- pima[-train_indices, ]

# Boosting
boost.train <- train_data
boost.test <- test_data
boost.fit <- gbm(classdigit ~ npregnant + glucose + diastolic.bp +
                   skinfold.thickness + bmi + pedigree + age, 
                 data = boost.train, n.trees = 1000, 
                 shrinkage = .1, interaction.depth = 3, distribution = "adaboost")

y_true_boost <- as.numeric(test_data$classdigit)-1
y_hat_boost <- predict(boost.fit, newdata = test_data, type = "response")
y_hat_boost <- ifelse(y_hat_boost > 0.5, 1, 0)-1
misclass_boost <- sum(abs(y_true_boost - y_hat_boost))/length(y_hat_boost)
misclass_boost # 0.33125

# Random Forest
rf.fit <- randomForest(classdigit ~ npregnant + glucose + diastolic.bp +
                         skinfold.thickness + bmi + pedigree + age, 
                       data = train_data, ntree = 100)

y_true_rf <- as.numeric(test_data$classdigit)-1
y_hat_rf <- predict(rf.fit, newdata = test_data, type = "response")
y_hat_rf <- as.numeric(y_hat_rf)-1
misclass_rf <- sum(abs(y_true_rf - y_hat_rf))/length(y_hat_rf)
misclass_rf # 0.25

# Single CART Model
model.control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
cart_model <- rpart(classdigit ~ npregnant + glucose + diastolic.bp +
                      skinfold.thickness + bmi + pedigree + age, 
                    data = train_data, method = "class", 
                    control = model.control)

# Compute test error for a single tree
my_pred_cart <- predict(cart_model, newdata = test_data, type = "class")
y_true_cart <- as.numeric(test_data$classdigit)-1 
y_hat_cart <- as.numeric(my_pred_cart)-1
misclass_cart <- sum(abs(y_true_cart - y_hat_cart))/length(y_hat_cart)
misclass_cart # 0.2875

# Partial Dependence Plots
# For Random Forest
x11()
varImpPlot(rf.fit)
importance(rf.fit)
summary(rf.fit)

# For Boosting
summary(boost.fit)

# For CART
x11()
plot(cart_model)
text(cart_model, use.n = TRUE, cex = .5)