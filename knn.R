library(readr)
library(class)

train <- read_csv("./data/input/train.csv")
test <- read_csv("./data/input/test.csv")

dim(train)
dim(test)

# kNN
set.seed(1234)
numTrain <- 30000
rows <- sample(1:nrow(train), numTrain)
#train.col.used <- 1:43
prediction <- knn(train = train[rows, -1], test = test, cl = train[rows, 1], k = 3)
prediction.table <- data.frame(ImageId = 1:nrow(test), Label = prediction)
write_csv(prediction.table, "./data/output/knn.csv")