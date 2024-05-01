#######################################################
# Homework 3 Question 4
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/19/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(ISLR2)
library(MASS)
library(class)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import and look at data
data(Weekly)
head(Weekly)

#######################################################
# Variable Information
# 1089 observations of 9 variables
# 1,089 weekly stock market returns for 21 years
# Year - integer, value from 1990 to 2010
# Lag1 - num
# Lag2 - num
# Lag3 - num
# Lag4 - num
# Lag5 - num
# Volume - num
# Today - num
# Direction - chr, "Up" or "Down"
#######################################################

# 1a - Produce some numerical and graphical summaries of 
# the “Weekly” data. Do there appear to be any patterns?

# Get a numerical summary of the data
summary(Weekly)

# Create pairwise scatterplots
pairs(Weekly)

# Investigate relationship of volume and Year
plot(Weekly$Year, Weekly$Volume)

# 1b - Use the full data to perform logistic regression 
# with “Direction” as the response and the five lag 
# variables, plus volume, as predictors. Use the summary 
# function to print the results. 

# Logistic Regression
full_model <- glm(Direction ~ Lag1 + Lag2 + Lag3 + 
                    Lag4 + Lag5 + Volume, 
                  data = Weekly, family = "binomial")

# Print Summary
summary(full_model)

# 1c - Fit the logistic model using a training data 
# period from 1990-2008, with “Lag2” as the only predictor. 
# Compute the confusion matrix, and the overall correct 
# fraction of predictions (aka misclassification rate) for 
# the held out data (that is, the data from 2009 and 2010).

# Create the training and test data subsets
train <- (Weekly$Year <= 2008)
test <- (Weekly$Year > 2008)

# Logistic Model with Training
training_model <- glm(Direction ~ Lag2, data = Weekly, 
                      subset = train, family = "binomial")
summary(training_model)

# Predict on the test data
train_prob = predict(training_model, Weekly[!train, ], type = "response")
train_pred = rep("Down", dim(Weekly[!train, ])[1])
train_pred[train_prob > 0.5] = "Up"

# Create a confusion matrix
conf <- table(train_pred, Weekly[!train, ]$Direction)
conf

# Compute misclassification rate
mean(train_pred == Weekly[!train, ]$Direction)

# 1d - Fit an LDA model using a training data period from 
# 1990-2008, with “Lag2” as the only predictor. Compute the 
# confusion matrix, and the overall correct fraction of 
# predictions (aka misclassification rate) for the held out 
# data (that is, the data from 2009 and 2010).

# Fit the LDA model
lda_model = lda(Direction ~ Lag2, data = Weekly, subset = train)
lda_model

# Predict on the test data
lda_pred = predict(lda_model, Weekly[!train, ])

# Create the confusion matrix
conf_lda <- table(lda_pred$class, Weekly[!train, ]$Direction)
conf_lda

# Compute misclassification rate
mean(lda_pred$class == Weekly[!train, ]$Direction)

# 1e - Fit a kNN model with k=1 using a training data period 
# from 1990-2008, with “Lag2” as the only predictor. Compute 
# the confusion matrix, and the overall correct fraction of 
# predictions (aka misclassification rate) for the held out 
# data (that is, the data from 2009 and 2010).

# Create test and training data frames
train_knn = data.frame(Weekly[train, ]$Lag2)
test_knn = data.frame(Weekly[!train, ]$Lag2)
train_knnD = Weekly[train, ]$Direction

# Fit the kNN model with k = 1
set.seed(123)
knn_pred <- knn(train_knn, test_knn, train_knnD, k = 1)
knn_pred

# Compute the confusion matrix
conf_knn <- table(knn_pred, Weekly[!train, ]$Direction)
conf_knn

# Compute the misclassification rate
mean(knn_pred == Weekly[!train, ]$Direction)