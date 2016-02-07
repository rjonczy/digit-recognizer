# Creates a simple random forest benchmark
library(randomForest)
library(readr)

set.seed(1234)

numTrain <- 10000
numTrees <- 25

train <- read_csv("./data/input/train.csv")
test <- read_csv("./data/input/test.csv")

rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows, 1])
train <- train[rows, -1]

rf <- randomForest(train, labels, xtest = test, ntree = numTrees)
predictions <- data.frame(ImageId = 1:nrow(test), Label = levels(labels)[rf$test$predicted])
head(predictions[,2])

write_csv(predictions, "./data/output/rf.csv") 

