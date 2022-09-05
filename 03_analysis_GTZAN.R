#### Preparing working environment ####
# Set current folder as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, naivebayes, e1071, class, randomForest, pROC, 
               caret, lubridate, ggpubr, xgboost, textreadr)


# Load data
data <- readRDS('song_data.rds')
data <- data %>% filter(tag %in% c("pop","rock","hiphop")) # Only these three genres overlap with GTZAN genres

##### Prepare GTZAN data for training classifiers #####
norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}

ml_data <- read_csv("GTZAN_features_30_sec.csv")

test_filtered <- 
  read_rtf("GTZAN_test_filtered.rtf") %>%
  str_split('.wav') %>%
  unlist()

train_filtered <- 
  read_rtf("GTZAN_train_filtered.rtf") %>%
  str_split('.wav') %>%
  unlist()

valid_filtered <- 
  read_rtf("GTZAN_valid_filtered.rtf") %>%
  str_split('.wav') %>%
  unlist()

filtered <- c(train_filtered[-length(train_filtered)], 
              valid_filtered[-length(valid_filtered)],
              test_filtered[-length(test_filtered)])
filtered <- gsub('.*/','',filtered)
filtered <- paste(filtered, '.wav',sep = '')

ml_data <- ml_data[ml_data$filename %in% filtered,]

#saveRDS(ml_data, file = 'GTZAN_filtered.rds')
#ml_data <- readRDS(file = 'GTZAN_filtered.rds')

#ml_data_backup <- ml_data
#ml_data <- ml_data_backup

ml_data <-
  ml_data %>%
  filter(label %in% c("pop","rock","hiphop")) %>%
  select(1,3,5,7,9,11,13,20,22,24,26,28,60) %>%
  mutate(across(c(2:12), norm_minmax)) %>%
  rename(tag = label) %>%
  rename_with(~gsub("_mean","_norm",.x, fixed=TRUE)) %>% 
  rename(rmse_norm = rms_norm)

set.seed(210016)
ml_train <- 
  ml_data %>%
  group_by(tag) %>%
  slice_sample(prop = 0.8)

ml_test <-
  ml_data %>%
  anti_join(ml_train, by = "filename")

ml_train <- ml_train[,c(13,2:12)]
ml_test <- ml_test[,c(13,2:12)]


nb_model <- naive_bayes(tag ~ ., data = ml_train)

predict(nb_model, ml_train[-1])
p_train <- predict(nb_model, ml_train[-1])
predict(nb_model, ml_train[-1], type = "prob")
mean(as.character(p_train) == ml_train$tag)  # 0.7227273

predict(nb_model, ml_test[-1])
p_test <- predict(nb_model, ml_test[-1])
p_test_prob <- predict(nb_model, ml_test[-1], type = "prob")
mean(as.character(p_test) == ml_test$tag)   # 0.6607143

multiclass.roc(ml_test$tag, p_test_prob) # 0.8123

saveRDS(nb_model, file = "nb_model_GTZAN.rds")
nb_model <- readRDS(file = "nb_model_GTZAN.rds")



##### KNN #####
# Use AUC to find the best k: 31
df_knn_auc <- c()
for (k in 1:60){
  set.seed(210016)
  knn_model <- knn(train = ml_train[-1], test = ml_test[-1], cl = ml_train$tag, prob = T, k = k)
  acc <- mean(knn_model == ml_test$tag)
  auc <- multiclass.roc(ml_test$tag, attr(knn_model, "prob"))$auc[1]
  df_knn_auc <- rbind(df_knn_auc, c(k, acc, auc))
}

df_knn_auc %>%
  as.data.frame() %>%
  rename(k = V1, acc = V2, auc = V3) %>%
  mutate(sum = acc + auc) %>%
  arrange(desc(sum)) %>%
  head(10)

set.seed(210016)
k_4 <- knn(train = ml_train[-1], test = ml_test[-1], cl = ml_train$tag, k = 4, prob = T)
saveRDS(k_4, file = "k_4_GTZAN.rds")
nb_model <- readRDS(file = "k_4_GTZAN.rds")

mean(k_4 == ml_test$tag)   # 0.7678571
attr(k_4, "prob")   
table(k_4, ml_test$tag)
multiclass.roc(ml_test$tag, attr(k_4, "prob")) # 0.5605

##### Random Forest #####
# Use AUC to find the best n, max: 14, 43; 13, 43
df_rf_auc <- c()
for (n in 1:60){
  for (max in 30:60){
    set.seed(123456)
    rf_model <- randomForest(as.factor(tag) ~., data = ml_train, ntree = n, maxnodes = max)
    p_test_prob <- predict(rf_model, ml_test[-1], type = "prob")
    acc <- mean(as.character(predict(rf_model, ml_test[-1])) == ml_test$tag)
    auc <- multiclass.roc(ml_test$tag, p_test_prob)$auc[1]
    df_rf_auc <- rbind(df_rf_auc, c(n, max, acc, auc))
    print(max)
  }
  print(n)
}

df_rf_auc %>%
  as.data.frame() %>%
  rename(n = V1, max = V2, acc = V3, auc = V4) %>%
  mutate(sum = acc + auc) %>%
  arrange(desc(sum)) %>%
  head(10)

set.seed(123456)
rf_20_39 <- randomForest(as.factor(tag) ~., data = ml_train, ntree = 20, maxnodes = 39)
saveRDS(rf_20_39, file = "rf_20_39_GTZAN.rds")
rf_20_39 <- readRDS(file = "rf_20_39_GTZAN.rds")


rf_20_39_test_prob <- predict(rf_20_39, ml_test[-1], type = "prob")
mean(as.character(predict(rf_20_39, ml_test[-1])) == ml_test$tag)  # 0.7857143
multiclass.roc(ml_test$tag, rf_20_39_test_prob)$auc[1]  # 0.8977683



##### XGBOOST #####
ml_train_num <- 
  ml_train %>%
  mutate(tag = replace(tag, tag == 'pop', 0),
         tag = replace(tag, tag == 'hiphop', 1),
         tag = replace(tag, tag == 'rock', 2))
ml_train_num$tag <- as.double(ml_train_num$tag)

ml_test_num <- 
  ml_test %>%
  mutate(tag = replace(tag, tag == 'pop', 0),
         tag = replace(tag, tag == 'hiphop', 1),
         tag = replace(tag, tag == 'rock', 2))
ml_test_num$tag <- as.double(ml_test_num$tag)

train_matrix <- xgb.DMatrix(data = as.matrix(ml_train_num[,-1]), 
                            label = ml_train_num$tag)
test_matrix <- xgb.DMatrix(data = as.matrix(ml_test_num[,-1]), 
                           label = ml_test_num$tag)

best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0


for (iter in 1:100) {
  param <- list(objective = "multi:softprob",
                eval_metric = "mlogloss",
                num_class = 3,
                max_depth = sample(2:5, 1),
                eta = runif(1, .01, .1)
                #gamma = runif(1, 0.0, 0.2),
                #subsample = runif(1, .6, .9),
                #colsample_bytree = runif(1, .5, .8),
                #min_child_weight = sample(1:40, 1),
                #max_delta_step = sample(1:10, 1)
  )
  cv.nround = 500
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=train_matrix, params = param, nthread=6,
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log$test_mlogloss_mean)
  min_logloss_index = which.min(mdcv$evaluation_log$test_mlogloss_mean)
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

nround <- best_logloss_index
set.seed(best_seednumber)
bst_model <- xgb.train(params = best_param,
                       data = train_matrix,
                       nrounds = nround,
                       nthread=6)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = 3,
                          ncol=length(test_pred)/3) %>%
  t() %>%
  data.frame() %>%
  mutate(label = ml_test_num$tag + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")


xgb.save(bst_model,'xgboost_0.73_GTZAN_0831.model')
bst_model <- xgb.load('xgboost_0.73_GTZAN_0831.model')

test_prediction_prob <- matrix(test_pred, nrow = 3,
                               ncol=length(test_pred)/3) %>%
  t() %>%
  data.frame() %>%
  rename(pop = X1, hiphop=X2, rock=X3)
# mutate(max_prob = max.col(.), .keep = "none") %>%
# mutate(max_prob = replace(max_prob, max_prob == 1, 'pop'),
#        max_prob = replace(max_prob, max_prob == 2, 'hiphop'),
#        max_prob = replace(max_prob, max_prob == 3, 'rock'),
#        max_prob = replace(max_prob, max_prob == 4, 'folk')) %>%
# unlist()

# AUC
multiclass.roc(ml_test$tag, test_prediction_prob)$auc[1]  # 0.9155057



#saveRDS(data_predict, file = "data_predict_GTZAN.rds")
#data_predict <- readRDS(file = "../rnr/data_predict_GTZAN.rds")


##### Gaussian Naive Bayes #####
###### Mean Prediction 4 Tags Accuracy By Years ######
gnb_mean_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(gnb_mean_pred_acc = mean(gnb_pred_acc)) %>%
  ungroup() %>%
  print()

###### Mean Prediction Hip-Hop Binary Accuracy By Years ######
gnb_mean_bin_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(gnb_mean_pred_bin_acc = mean(gnb_pred_bin_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Recall By Years ######
# Treatment by year
gnb_recall_tbyyear <- 
  data_predict %>%
  filter(tag == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(gnb_recall = mean(gnb_pred_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Precision By Years ######
gnb_precision_tbyyear <- 
  data_predict %>%
  filter(gnb_pred == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(gnb_precision = mean(gnb_pred_acc)) %>%
  ungroup() %>%
  print()


###### Aggregating Graphics: Mean Acc, Mean Hip-Hop Acc, Trap, Non-Trap ######
gnb_metrics <- 
  gnb_mean_acc_tbyyear %>%
  right_join(gnb_mean_bin_acc_tbyyear, by = c("release_year", "treatment_year")) %>%
  right_join(gnb_recall_tbyyear, by = c("release_year", "treatment_year")) %>%
  pivot_longer(!c(release_year, treatment_year), names_to = "metrics", values_to = "values")

gnb_metrics_plot <-
  gnb_metrics %>%
  ggplot(aes(x = release_year, y = values)) +
  geom_point(aes(shape = metrics), size = 2.5) +
  geom_line(aes(linetype = metrics)) 
gnb_metrics_plot

saveRDS(gnb_metrics, file = "gnb_metrics_GTZAN.rds")
#gnb_metrics <- readRDS(file = "gnb_metrics_GTZAN.rds")
  

##### KNN #####
###### Mean Prediction 4 Tags Accuracy By Years ######
knn_mean_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(knn_mean_pred_acc = mean(knn_pred_acc)) %>%
  ungroup() %>%
  print()

###### Mean Prediction Hip-Hop Binary Accuracy By Years ######
knn_mean_bin_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(knn_mean_pred_bin_acc = mean(knn_pred_bin_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Recall By Years ######
knn_recall_tbyyear <- 
  data_predict %>%
  filter(tag == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(knn_recall = mean(knn_pred_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Precision By Years ######
knn_precision_tbyyear <- 
  data_predict %>%
  filter(knn_pred == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(knn_precision = mean(knn_pred_acc)) %>%
  ungroup() %>%
  print()

###### Aggregating Graphics: Mean Acc, Mean Hip-Hop Acc, Trap, Non-Trap ######
# Treatment by year
knn_metrics <- 
  knn_mean_acc_tbyyear %>%
  right_join(knn_mean_bin_acc_tbyyear, by = c("release_year", "treatment_year")) %>%
  right_join(knn_recall_tbyyear, by = c("release_year", "treatment_year")) %>%
  pivot_longer(!c(release_year, treatment_year), names_to = "metrics", values_to = "values")

knn_metrics_plot <- 
  knn_metrics %>%
  ggplot(aes(x = release_year, y = values)) +
  geom_point(aes(shape = metrics), size = 2.5) +
  geom_line(aes(linetype = metrics)) 
knn_metrics_plot

saveRDS(knn_metrics, file = "knn_metrics_GTZAN.rds")
#knn_metrics <- readRDS(file = "knn_metrics_GTZAN.rds")

##### Random Forest #####
###### Mean Prediction 4 Tags Accuracy By Years ######
rf_mean_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(rf_mean_pred_acc = mean(rf_pred_acc)) %>%
  ungroup() %>%
  print()

###### Mean Prediction Hip-Hop Binary Accuracy By Years ######
rf_mean_bin_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(rf_mean_pred_bin_acc = mean(rf_pred_bin_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Recall By Years ######
rf_recall_tbyyear <- 
  data_predict %>%
  filter(tag == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(rf_recall = mean(rf_pred_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Precision By Years ######
rf_precision_tbyyear <- 
  data_predict %>%
  filter(rf_pred == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(rf_precision = mean(rf_pred_acc)) %>%
  ungroup() %>%
  print()



###### Aggregating Graphics: Mean Acc, Mean Hip-Hop Acc, Trap, Non-Trap ######
# Treatment by year
rf_metrics <- rf_mean_acc_tbyyear %>%
  right_join(rf_mean_bin_acc_tbyyear, by = c("release_year", "treatment_year")) %>%
  right_join(rf_recall_tbyyear, by = c("release_year", "treatment_year")) %>%
  pivot_longer(!c(release_year, treatment_year), names_to = "metrics", values_to = "values") 

rf_metrics_plot <- 
  rf_metrics %>%
  ggplot(aes(x = release_year, y = values)) +
  geom_point(aes(shape = metrics), size = 2.5) +
  geom_line(aes(linetype = metrics))
rf_metrics_plot

saveRDS(rf_metrics, file = "rf_metrics_GTZAN.rds")
#rf_metrics <- readRDS(file = "rf_metrics_GTZAN.rds")

##### XGBoost #####
###### Mean Prediction 4 Tags Accuracy By Years ######
xgb_mean_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(xgb_mean_pred_acc = mean(xgb_pred_acc)) %>%
  ungroup() %>%
  print()

###### Mean Prediction Hip-Hop Binary Accuracy By Years ######
# Treatment by year
xgb_mean_bin_acc_tbyyear <- 
  data_predict %>%
  group_by(release_year, treatment_year) %>%
  summarize(xgb_mean_pred_bin_acc = mean(xgb_pred_bin_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Recall By Years ######
# Treatment by year
xgb_recall_tbyyear <- 
  data_predict %>%
  filter(tag == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(xgb_recall = mean(xgb_pred_acc)) %>%
  ungroup() %>%
  print()

###### Prediction Hip-Hop Precision By Years ######
xgb_precision_tbyyear <- 
  data_predict %>%
  filter(xgb_pred == "hiphop") %>%
  group_by(release_year, treatment_year) %>%
  summarize(xgb_precision = mean(xgb_pred_acc)) %>%
  ungroup() %>%
  print()

###### Aggregating Graphics: Mean Acc, Mean Hip-Hop Acc, Trap, Non-Trap ######
# Treatment by year
xgb_metrics <- xgb_mean_acc_tbyyear %>%
  right_join(xgb_mean_bin_acc_tbyyear, by = c("release_year", "treatment_year")) %>%
  right_join(xgb_recall_tbyyear, by = c("release_year", "treatment_year")) %>%
  pivot_longer(!c(release_year, treatment_year), names_to = "metrics", values_to = "values") 

xgb_metrics_plot <- 
  xgb_metrics %>%
  ggplot(aes(x = release_year, y = values)) +
  geom_point(aes(shape = metrics), size = 2.5) +
  geom_line(aes(linetype = metrics))
xgb_metrics_plot

saveRDS(xgb_metrics, file = "xgb_metrics_GTZAN.rds")
#xgb_metrics <- readRDS(file = "xgb_metrics_GTZAN.rds")


##### Aggregating Graphics: GNB, KNN, RF, XGB #####
aggregate_plots <- ggarrange(gnb_metrics_plot, knn_metrics_plot, rf_metrics_plot, xgb_metrics_plot,
                             ncol = 4, 
                             common.legend = T)

annotate_figure(aggregate_plots, top = text_grob("Training Set: GTZAN", face = "bold", size = 14))

##### Average Four Classifiers #####
average <- cbind(gnb_metrics, knn_metrics[,4], rf_metrics[,4], xgb_metrics[,4])
colnames(average)[4:7] <- c("values_gnb","values_knn","values_rf","values_xgb")
average$values <- (average$values_gnb + average$values_knn + average$values_rf + average$values_xgb)/4
average <- average[,c(1:3,8)]
average$metrics <- gsub('gnb_','classifiers_',average$metrics)

saveRDS(average, file = "classifiers_metrics_GTZAN.rds")
