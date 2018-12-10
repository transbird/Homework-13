library(dplyr)
library(caret)
library(nnet)

#TRAIN DATASET
# get the training datasets
if (!exists("mtrain")) {
  mtrain <- read.csv("mnist_train.csv", header=F) %>% as.matrix
  train_classification <- mtrain[,1]  # y value
  
  mtrain <- mtrain[,-1]/256  # x matrix
  colnames(mtrain) <- 1:(28^2)
  x <- mtrain[1:1000, ]
  
  rownames(mtrain) <- NULL
}

y <- rep(NA, length(train_classification))

#Changing all 3s to 1 and all other numbers to 0
for (i in 1:length(train_classification)){ 
  i_th <- train_classification[i]
  
  if (i_th==3){ 
      i_th <- 1
    } else { 
      i_th <- 0
    }
  y[i] <- i_th 
}

# for caret, y variable should be a factor
# see line 54 in caret_intro_2d.R
y <- factor(y, levels=c(0,1)) 
y <- y[1:1000]


#TEST DATASET
#getting the test data set
if (!exists("mtrain_t")) {
  mtrain_t <- read.csv("mnist_test.csv", header=F) %>% as.matrix
  train_classification_t <- mtrain_t[,1]  # y value
  
  mtrain_t <- mtrain_t[,-1]/256  # x matrix
  colnames(mtrain_t) <- 1:(28^2)
  x_t <- mtrain_t[1:1000, ]
  
  rownames(mtrain_t) <- NULL
}

y_t <- rep(NA, length(train_classification_t))

#Changing all 3s to 1 and all other numbers to 0
for (i in 1:length(train_classification_t)){ 
  i_th <- train_classification_t[i]
  
  if (i_th==3){ 
    i_th <- 1
  } else { 
    i_th <- 0
  }
  y_t[i] <- i_th 
}

# for caret, y variable should be a factor
# see line 54 in caret_intro_2d.R
y_t <- factor(y_t, levels=c(0,1)) 
y_t <- y_t[1:1000]

head(y)
tail(y)
head(y_t)
tail(y_t)

#NEURAL NET
# let's try and fit with a neural net. 
#The process of training

#With Fixed decay = 0
tuning_df <- data.frame(size=1, decay=0)
#fitControl <- trainControl(method="none")
fitControl <- trainControl(## 2-fold CV 
  method = "repeatedcv",
  number = 2,
  repeats = 2)
t_out <- caret::train(x=x, y=y, method="nnet", 
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=1000, MaxNWts=10000)

#Predicting the errors
#Write a predicting-errors function first
predicting_errors <- function(y, t_out)
{
  true_y <- y
  predict_y <- predict(t_out, x)
  n_samples <- nrow(x)
  error <- sum(true_y != predict_y)/n_samples
  return(error)
}
#Use the function
predict_error <- predicting_errors(y, t_out)
cat("train prediction error", predict_error, "\n")

#To find the best size, we can use a for loop
error_list <- rep(NA, 5)
for (i in 1:5) {
  tuning_df <- data.frame(size=i, decay=0)
  fitControl <- trainControl(## 2-fold CV 
    method = "repeatedcv",
    number = 2,
    repeats = 2)
  t_out <- caret::train(x=x, y=y, method="nnet", 
                        trControl = fitControl,
                        tuneGrid=tuning_df, maxit=1000, MaxNWts=10000)
  predict_error <- predicting_errors(y, t_out)
  error_list[i] <- predict_error
  
}
error_list
#We decide that size=3, erros is 0, a good enough nod



#With Varied decay
tuning_df <- data.frame(size=3, decay=.1)
fitControl <- trainControl(method="none")
fitControl <- trainControl(## 2-fold CV 
  method = "repeatedcv",
  number = 2,
  repeats = 2)
t_out_1 <- caret::train(x=x, y=y, method="nnet", 
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=1000, MaxNWts=10000)

#Predicting the errors
predict_error <- predicting_errors(y, t_out_1)
cat("train1 prediction error", predict_error, "\n")

#To find the best decay, we can use a for loop
error_list_d <- rep(NA, 5)
for (i in 0:4) {
  tuning_df <- data.frame(size=3, decay=i/10)
  fitControl <- trainControl(## 2-fold CV 
    method = "repeatedcv",
    number = 2,
    repeats = 2)
  t_out_1 <- caret::train(x=x, y=y, method="nnet", 
                        trControl = fitControl,
                        tuneGrid=tuning_df, maxit=1000, MaxNWts=10000)
  predict_error <- predicting_errors(y, t_out_1)
  error_list_d[i] <- predict_error
  
}
error_list_d


#We decide Size=3 decay=0.1 gives the best result

#CROSS VALIDATION TEST
tuning_df <- data.frame(size=3, decay=.1)
fitControl <- trainControl(method="none")
fitControl <- trainControl(## 2-fold CV 
  method = "repeatedcv",
  number = 2,
  repeats = 2)
t_out_f <- caret::train(x=x, y=y, method="nnet", 
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=1000, MaxNWts=10000)

#Predicting the errors
predict_error <- predicting_errors(y, t_out_f)
cat("train1.1 prediction error", predict_error, "\n")

#Compare to the test data errors
pred_error_t <- predicting_errors(y_t, t_out_f)
cat("test prediction error", pred_error_t, "\n")





