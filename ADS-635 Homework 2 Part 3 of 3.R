#######################################################
# Homework 2 Question 3
# ADS-635: Data Mining I
# Alison Croteau
# Created: 9/28/2023
# Modified: ---
#######################################################

# Install and import necessary libraries
library(class)
library(tidyverse)

# Initialize workspace
rm(list = ls())
setwd("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635")

# Import data
zip.train <- read.table("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\zip.train",
                        sep = " ")
zip.test <- read.table("C:\\Users\\acrot\\OneDrive\\Documents\\ADS-635\\zip.test",
                       sep = " ")

#######################################################
# Variable Information
# Normalized handwritten digits, automatically
# scanned from envelopes.
# The data are in two gzipped files, and each line consists of 
# the digit id (0-9) followed by the 256 gray scale values.
#######################################################

# Remove last variable column from training dataset
zip.train <- subset(zip.train, select = -V258)

# Subset the data to only include 0's and 8's
train = zip.train %>% filter(V1 == 0 | V1 == 8)
test = zip.test %>% filter(V1 == 0 | V1 == 8)

# k = 1
knn.pred.k1 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 1)
table(knn.pred.k1, test$V1)
k1.error <- mean(knn.pred.k1 != test$V1)
k1.error

# k = 3
knn.pred.k3 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 3)
table(knn.pred.k3, test$V1)
k3.error <- mean(knn.pred.k3 != test$V1)
k3.error

# k = 5
knn.pred.k5 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 5)
table(knn.pred.k5, test$V1)
k5.error <- mean(knn.pred.k5 != test$V1)
k5.error

# k = 7
knn.pred.k7 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 7)
table(knn.pred.k7, test$V1)
k7.error <- mean(knn.pred.k7 != test$V1)
k7.error

# k = 9
knn.pred.k9 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 9)
table(knn.pred.k9, test$V1)
k9.error <- mean(knn.pred.k9 != test$V1)
k9.error

# k = 11
knn.pred.k11 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 11)
table(knn.pred.k11, test$V1)
k11.error <- mean(knn.pred.k11 != test$V1)
k11.error

# k = 13
knn.pred.k13 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 13)
table(knn.pred.k13, test$V1)
k13.error <- mean(knn.pred.k13 != test$V1)
k13.error

# k = 15
knn.pred.k15 <- knn(train = train[, -1], test = test[, -1],  cl = train$V1, k = 15)
table(knn.pred.k15, test$V1)
k15.error <- mean(knn.pred.k15 != test$V1)
k15.error

# Linear Regression
lm.fit <- lm(V1 ~ ., data = train)
summary(lm.fit)
lm.pred <- predict(lm.fit, newdata = test)

# Convert regression predictions to classification
lm.class.pred <- ifelse(lm.pred < 4, 0, 8)

# Compute test error for linear regression
lm.error <- mean(lm.class.pred != test$V1)
lm.error