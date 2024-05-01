#######################################################
# Homework 3 Question 1
# ADS-635: Data Mining I
# Alison Croteau
# Created: 10/15/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
# install.packages("heplots")
library(heplots)
library(MASS)
attach(Diabetes)
library(e1071)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import data
data(Diabetes)
head(Diabetes)

#######################################################
# Variable Information
# 145 observations of 6 variables
# relwt - relative weight, expressed as the ratio of 
# actual weight to expected weight, given the person's 
# height, a numeric vector
#
# glufast - fasting plasma glucose level, a numeric vector
#
# glutest - test plasma glucose level, a measure of 
# glucose intolerance, a numeric vector
#
# instest - plasma insulin during test, a measure of 
# insulin response to oral glucose, a numeric vector
#
# sspg - steady state plasma glucose, a measure of 
# insulin resistance, a numeric vector
#
# group - diagnostic group, a factor with levels 
# Normal, Chemical_Diabetic, Overt_Diabetic
#######################################################

# 1a - Produce pairwise scatter plots for all five variables, 
# with different symbols or colors representing the three different 
# classes. Do you see any evidence that the classes may have 
# different covariance matrices? That they may not be multivariate normal?

colors <- c("red", "green", "blue") 
pairs(Diabetes[,1:5], 
      col = colors[Diabetes$group],
      pch = 20,
      main = "Pairwise Scatterplots")

# 1b - Apply linear discriminant analysis (LDA) and quadratic 
# discriminant analysis (QDA). How does the performance of QDA 
# compare to that of LDA in this case?

# Split the data into test and training
set.seed(123) # Setting seed for reproducibility
sample_index <- sample(1:nrow(Diabetes), nrow(Diabetes)*0.7)
train_data <- Diabetes[sample_index, ]
test_data <- Diabetes[-sample_index, ]

# Apply LDA and QDA
lda_model <- lda(group ~ ., data = train_data)
qda_model <- qda(group ~ ., data = train_data)
lda_model
qda_model

# Model predictions of class labels for test data
lda_pred <- predict(lda_model, test_data)
names(lda_pred)
qda_pred <- predict(qda_model, test_data)
names(qda_pred)

lda_class <- lda_pred$class
table(lda_class, test_data$group)
qda_class <- qda_pred$class
table(qda_class, test_data$group)

# Compare the accuracy of the predictions
lda_acc <- sum(lda_class == test_data$group) / nrow(test_data)
qda_acc <- sum(qda_class == test_data$group) / nrow(test_data)
lda_acc
qda_acc

# 1c - Suppose an individual has (glucose test/intolerance = 
# 168, instest = 122, SSPG = 541, Relative weight = .86, fasting 
# plasma glucose = 184). To which class does LDA assign this 
# individual? To which class does QDA?

# Create new individual data
new_data <- data.frame(relwt = 0.86, glufast = 184, glutest = 168, 
  instest = 122, sspg = 541)

# Predict the class labels for the given individual
lda_prediction <- predict(lda_model, new_data)$class
qda_prediction <- predict(qda_model, new_data)$class
lda_prediction
qda_prediction