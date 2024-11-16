# Load the necessary packages
library(raster)
library(plyr)
library(dplyr)
library(caret)
library(randomForest)
library(nnet)
library(NeuralNetTools)
library(parallel)
library(doParallel)
library(ggplot2)

# Setting the directory for caching rasters
rasterOptions(tmpdir = "./temp")

# Loading rasters
PC1 = raster("PC1.tif")
PC2 = raster("PC2.tif")
PC3 = raster("PC3.tif")
PC4 = raster("PC4.tif")
PC5 = raster("PC5.tif")
PC6 = raster("PC6.tif")

# Normalizing raster layers
PC1s <- scale(PC1)
PC2s <- scale(PC2)
PC3s <- scale(PC3)
PC4s <- scale(PC4)
PC5s <- scale(PC5)
PC6s <- scale(PC6)


# Combining rasters into a stack
ImageStack <- stack(PC1s, PC2s, PC3s, PC4s, PC5s, PC6s)

# Renaming rasters in the stack
names(ImageStack) <- c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6")

# Removing unnecessary objects
rm(PC1, PC2, PC3, PC4, PC5, PC6, PC1s, PC2s, PC3s, PC4s, PC5s, PC6s)

# Loading a shapefile with polygons of reference classes
trainData <- shapefile("data/TrainTestSamplePreparing.shp")

# Create an integer field with class numbers
trainData@data$ClassInt <- as.integer(factor(trainData@data$C_name))

# Number of available training attributes by class
summary(factor(trainData@data$C_name))

# Extraction of pixel values within the reference sample polygons into a table
dfAll <- data.frame(matrix(vector(), nrow = 0, ncol = length(names(ImageStack)) + 1))
for (i in 1:length(unique(trainData[["ClassInt"]]))){
  category <- unique(trainData[["ClassInt"]])[i]
  categorymap <- trainData[trainData[["ClassInt"]] == category,]
  dataSet <- extract(ImageStack, categorymap)
  dataSet <- dataSet[!unlist(lapply(dataSet, is.null))]
  dataSet <- lapply(dataSet, function(x){cbind(x, ClassInt = as.numeric(rep(category, nrow(x))))})
  df <- do.call("rbind", dataSet)
  dfAll <- rbind(dfAll, df)
}

# Returning symbolic class names to the table obtained in the previous step
ClassDF <- data.frame(Class = levels(factor(trainData@data$C_name)), ClassInt = seq(1:11))
dfAll <- left_join(dfAll, ClassDF)


# Preparing a table with reference data
LearnDF <- select(dfAll, -ClassInt)

# Print the number of pixels of each class in the resulting table
summary(factor(LearnDF$Class))

# Dividing the table with reference data into training (70%) and test (30%) samples
set.seed(456)
idx_train <- createDataPartition(LearnDF$Class, p = 0.7, list = FALSE)
dt_train <- LearnDF[idx_train,]
dt_test <- LearnDF[-idx_train,]

# Number of pixels in the training set by class
table(dt_train$Class)
# Number of pixels in the test sample by class
table(dt_test$Class)

# Setting up cross-validation of models using the k-block method
n_folds <- 10
set.seed(456)
folds <- createFolds(1:nrow(dt_train), k = n_folds)
seeds <- vector(mode = "list", length = n_folds + 1)
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1)

# Creating a variable that will contain a number of machine learning parameters 
# that will be used in all classification methods
ctrl <- trainControl(summaryFunction = multiClassSummary,
                     method = "cv",
                     number = n_folds,
                     search = "grid",
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     index = folds,
                     seeds = seeds)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classification of remote sensing data by the random forest method
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Creating a cluster for parallel computing using 3/4 
# of the number of processors
cl <- makeCluster(3/4 * detectCores())
registerDoParallel(cl)

# Model training
model_rf <- caret::train(Class ~ . , method = "rf",
  data = dt_train,
                 importance = TRUE,
                 allowParallel = TRUE,
                 tuneGrid = data.frame(mtry = c(2, 3, 4, 5, 6)),
                 trControl = ctrl)

# Stopping and deleting a parallel computing cluster
stopCluster(cl); remove(cl)

registerDoSEQ()

# Saving the resulting model to a file
saveRDS(model_rf, file = "models/model_rf.rds")

# Output of estimated model parameters
# Total time of model calculation
model_rf$times$everything

# Prints the result of the model setup
ggplot(model_rf)

# Calculation of the confusion matrix and other model estimates on test data
cm_rf <- confusionMatrix(data = predict(model_rf, newdata = dt_test),
                         factor(dt_test$Class))
cm_rf


randomForest::varImpPlot(model_rf$finalModel)

# Creation of a predictive thematic map based on the obtained model
predict_rf <- raster::predict(object = ImageStack,
                              model = model_rf, type = 'raw')

# Saving the resulting thematic map to a file in GeoTIFF format
writeRaster(predict_rf, "results/RFPredictMap.tif")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classification of remote sensing by the method of support vectors
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Creating a set of combinations of model hyperparameters
svm_grid <- expand.grid(cost = c(0.2, 0.5, 1),
                        Loss = c("L1", "L2"))

# Creating a cluster for parallel computing using 3/4 
# of the number of processors
cl <- makeCluster(3/4 * detectCores())
registerDoParallel(cl)

# Model training
model_svm <- caret::train(Class ~ . , method = "svmLinear3", data = dt_train,
                          allowParallel = TRUE,
                          tuneGrid = svm_grid,
                          trControl = ctrl)

# Stopping and deleting a parallel computing cluster
stopCluster(cl); remove(cl)

registerDoSEQ()

# Saving the resulting model to a file
saveRDS(model_svm, file = "models/model_svm.rds")

# Output of estimated model parameters
# Total time of model calculation
model_svm$times$everything

# Prints the result of the model setup
ggplot(model_svm)

# Calculation of the confusion matrix and other model estimates on test data
cm_svm <- confusionMatrix(data = predict(model_svm, newdata = dt_test),
                          factor(dt_test$Class))

# Calculation of the confusion matrix and other model estimates on test data
cm_svm

# Creation of a predictive thematic map based on the obtained model
predict_svm <- raster::predict(object = ImageStack,
                               model = model_svm, type = 'raw')

# Saving the resulting thematic map to a file in GeoTIFF format
writeRaster(predict_svm, "results/SVMPredictMap.tif")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classification of remote sensing data using an artificial neural network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Creating a set of combinations of model hyperparameters
nnet_grid <- expand.grid(size = c(5, 10, 15),
                         decay = c(0.001, 0.01, 0.1))

# Creating a cluster for parallel computing using 3/4 
# of the number of processors
cl <- makeCluster(3/4 * detectCores())
registerDoParallel(cl)

# Model training
model_nnet <- train(Class ~ ., method = 'nnet', data = dt_train,
                    importance = TRUE,
                    maxit = 1000,
                    allowParallel = TRUE,
                    tuneGrid = nnet_grid,
                    trControl = ctrl)

# Stopping and deleting a parallel computing cluster
stopCluster(cl); remove(cl)

registerDoSEQ()

# Saving the resulting model to a file
saveRDS(model_nnet, file = "models/model_nnet.rds")


# Output of estimated model parameters
# Total time of model calculation
model_nnet$times$everything


model_nnet$finalModel

# Prints the result of the model setup
ggplot(model_nnet)

# Print an illustration of an artificial neural network model architecture
plotnet(model_nnet)

# Calculation of the confusion matrix and other model estimates on test data
cm_nnet <- confusionMatrix(data = predict(model_nnet,
newdata = dt_test),
               factor(dt_test$Class))

cm_nnet

# Creation of a predictive thematic map based on the obtained model
predict_nnet <- raster::predict(object = ImageStack,
                                model = model_nnet, type = 'raw')

# Saving the resulting thematic map to a file in GeoTIFF format
writeRaster(predict_nnet, "results/NnetPredictMap.tif")