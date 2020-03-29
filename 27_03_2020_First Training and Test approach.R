#### machine learning:D####

# 27.03.2020
rm(list=ls())
setwd("C:/Users/David/Desktop/Ordner/Projekte/Machine Learning")
library(dataPreparation)

data("adult")
print(head(adult, n = 4))

#preparing data####
# splitting training and test data
# Random sample indexes
train_index <- sample(1:nrow(adult), 0.8 * nrow(adult))
test_index <- setdiff(1:nrow(adult), train_index)


# Build X_train, y_train, X_test, y_test
X_train <- adult[train_index, -15]
y_train <- adult[train_index, "income"]

X_test <- adult[test_index, -15]
y_test <- adult[test_index, "income"]


# filter useless variables
constant_cols <- whichAreConstant(adult)

# [1] "whichAreConstant: it took me 0s to identify 0 constant column(s)"
double_cols <- whichAreInDouble(adult)
# [1] "whichAreInDouble: it took me 0s to identify 0 column(s) to drop."
bijections_cols <- whichAreBijection(adult)
# [1] "whichAreBijection: education_num is a bijection of education. I put it in drop list."
# [1] "whichAreBijection: it took me 0.07s to identify 1 column(s) to drop."

X_train$education_num = NULL
X_test$education_num = NULL

#scaling data
scales <- build_scales(dataSet = X_train, cols = c("capital_gain", "capital_loss"), verbose = TRUE)
# [1] "build_scales: I will compute scale on  2 numeric columns."
# [1] "build_scales: it took me: 0s to compute scale for 2 numeric columns."
print(scales)

X_train <- fastScale(dataSet = X_train, scales = scales, verbose = TRUE)
X_test <- fastScale(dataSet = X_test, scales = scales, verbose = TRUE)

print(head(X_train[, c("capital_gain", "capital_loss")]))

#discretization of data
bins <- build_bins(dataSet = X_train, cols = "age", n_bins = 10, type = "equal_freq")
print(bins)
X_train <- fastDiscretization(dataSet = X_train, bins = list(age = c(0, 18, 25, 45, 62, +Inf)))
X_test <- fastDiscretization(dataSet = X_test, bins = list(age = c(0, 18, 25, 45, 62, +Inf)))
print(table(X_train$age))

#encoding categorical
encoding <- build_encoding(dataSet = X_train, cols = "auto", verbose = TRUE)
X_train <- one_hot_encoder(dataSet = X_train, encoding = encoding, drop = TRUE, verbose = TRUE)
X_test <- one_hot_encoder(dataSet = X_test, encoding = encoding, drop = TRUE, verbose = TRUE)
print("Dimensions of X_train: ")
print(dim(X_train))
print("Dimensions of X_test: ")
print(dim(X_test))

# filtering variables
bijections <- whichAreBijection(dataSet = X_train, verbose = TRUE)
X_train$sex.Male = NULL
X_test$sex.Male = NULL

#3 controlling shape ####
X_test <- sameShape(X_train, referenceSet = X_test, verbose = TRUE)


#done

#29.03.2020 start a new one:) https://www.r-bloggers.com/how-to-prepare-and-apply-machine-learning-to-your-dataset/####
rm(list=ls())



