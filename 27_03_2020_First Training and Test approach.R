#### machine learning:D####

# 27.03.2020 (https://cran.r-project.org/web/packages/dataPreparation/vignettes/train_test_prep.html)
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
library(caret) #for machine learning

#dataset loading and preperation####
data(iris)

#create a list of 80% of rows in the original dataset to use them for training
validation_index <- createDataPartition(iris$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- iris[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- iris[validation_index,]

#dataset summary, know your data:)####
#the dimension
dim(dataset)
#the variables and their character(class)
sapply(dataset, class)
#how does it actually look like?
head(dataset)
#which levels are within the factors (multinomial or binary?)
levels(dataset$Species)
#observations per species (level occurance, class distribution)
percentage <- prop.table(table(dataset$Species))*100
cbind(freq=table(dataset$Species),percentage=percentage)
#Statistical summary
summary(dataset)

#dataset visualization

#univariate plots (x(y))
x <- dataset[,1:4]
y <- dataset[,5]

par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i],main=names(iris)[i])
}
dev.off()
plot(y)

#multivariate plots
install.packages("ellipse")
library(ellipse)
featurePlot(x=x, y=y, plot="ellipse")
featurePlot(x=x, y=y, plot="box")

scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Algorthms evaluation####

# Now it is time to create some models of the data and estimate their accuracy on unseen data.
#1 use the test harness to use 10-fold cross validation (cv)
#2 build 5 different models to predict species from flower measurements
#3 select the best model

#test harness
control <- trainControl(method="cv", number=10)
metric <- "Accuracy" # accuracy is the number of correctly predicted instances divided by the total number of instances in the dataset (times 100 for percentage)

# Build models
#we don't know yet which algorithms would be good on this problem
#1 linear discriminant analysis (LDA)
#2 classification and regression trees (CART)
#3 k-Nearest Neighbours (kNN)
#4 Support Vector Machines (SVM) with linear kernel
#5 Random Forest (RF)

#--> mixture of simple linear (LDA), nonlinear (CART, kNN), and complex nonlinear (SVM, RF) methods ()
library(kernlab)
library(e1071)
library(randomForest)

#let's go
#a) linear algorithms
set.seed(7)
fit.lda <- train(Species~.,data=dataset, method="lda",
                 metric=metric, trControl=control)
#b) nonlinear algorithms
#CART
set.seed(7)
fit.cart <- train(Species~.,data=dataset, method="rpart",
                  metric=metric, trControl=control)
#kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn",
                 metric=metric, trControl=control)
#c) advanced algorithms
#SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial",
                 metric=metric, trControl=control)
#RF
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf",
                metric=metric, trControl=control)

#Select best model

results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
dotplot(results)
print(fit.lda)

#test the LDA model on the validation data split

predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)


