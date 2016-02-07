library(readr)
library(class)

train <- read_csv("./data/input/train.csv")
test <- read_csv("./data/input/test.csv")

dim(train)
dim(test)

# PCA
train.columns.var <- apply(train[,-1], 2, var)
table(train.columns.var != 0)
train.zeroVarRemoved <- train[ ,c(F, train.columns.var != 0)]
pca.result <- prcomp(train.zeroVarRemoved, scale=T)
train.pca <- pca.result$x
test.pca <- predict(pca.result, test)

# kNN
set.seed(0)
numTrain <- 30000
rows <- sample(1:nrow(train), numTrain)
train.col.used <- 1:43
prediction <- knn(train.pca[rows,train.col.used], test.pca[,train.col.used], train[rows,1], k=3)
prediction.table <- data.frame(ImageId = 1:nrow(test), Label = prediction)
write_csv(prediction.table, "./data/output/pca_knn.csv")

