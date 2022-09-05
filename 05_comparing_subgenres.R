#### Preparing working environment ####
# Set current folder as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, ggpubr)

# Import predicted data
data_predict <- readRDS(file = "data_predict_2009.rds")  # Change for checking different year cohort classifiers

# One-hot coding subgenres
subgenres <-
  str_split(data_predict$genre, "#") %>%
  map(as.data.frame) %>%
  bind_rows(.id = "id") %>%
  rename(genre = 2) %>%
  group_by(id,genre) %>%
  summarize(n=n()) %>%
  spread(genre, n) %>% 
  arrange(as.double(id))
#subgenres[is.na(subgenres)] <- 0

data_predict_subgenres <- bind_cols(data_predict,subgenres) %>% rename(na = last_col())

#### Check overall prediction accuracy ####
data_predict_acc <- 
  data_predict %>%
  group_by(release_year) %>%
  summarize(gnb_mean = mean(gnb_pred_acc),
            gnb_bin_mean = mean(gnb_pred_bin_acc),
            knn_mean = mean(knn_pred_acc),
            knn_bin_mean = mean(knn_pred_bin_acc),
            rf_mean = mean(rf_pred_acc),
            rf_bin_mean = mean(rf_pred_bin_acc),
            xgb_mean = mean(xgb_pred_acc),
            xgb_bin_mean = mean(xgb_pred_bin_acc),
            classifiers_mean = mean(c(gnb_mean, knn_mean, rf_mean, xgb_mean)),
            classifiers_bin_mean = mean(c(gnb_bin_mean, knn_bin_mean, rf_bin_mean, xgb_mean))) %>% print()

data_predict_recall <- 
  data_predict %>%
  filter(tag == "hiphop") %>%
  group_by(release_year) %>%
  summarize(gnb_recall = mean(gnb_pred_acc),
            knn_recall = mean(knn_pred_acc),
            rf_recall = mean(rf_pred_acc),
            xgb_recall = mean(xgb_pred_acc),
            classifiers_recall = mean(c(gnb_recall, knn_recall, rf_recall, xgb_recall))) %>% print()

#### Check performance metrics of top 10 subgenre tags within Hip-Hop ####
# Check overall performance
df_subg <- 
  data_predict_subgenres %>%
  #filter(release_year == 2018) %>%
  filter(tag == "hiphop") %>% 
  #select(mp3_file_name:id | where(~is.numeric(.x) && sum(.x)>0)) %>%
  select(where(~!all(is.na(.x)))) %>%
    { map_dfr(.[,105:ncol(.)], 
            function(x) aggregate(list(gnb_bin_mean = .[["gnb_pred_bin_acc"]],
                                       knn_bin_mean = .[["knn_pred_bin_acc"]],
                                       rf_bin_mean = .[["rf_pred_bin_acc"]], 
                                       xgb_bin_mean = .[["xgb_pred_bin_acc"]]),
                                  list(group = x), 
                                  function(y) c(mean(y), length(y))),
            .id = "subgenre") } %>% 
  mutate(mean = (gnb_bin_mean + knn_bin_mean + rf_bin_mean + xgb_bin_mean)/4) %>%
  mutate(prod = mean[,1] * mean[,2],
         prod_gnb = gnb_bin_mean[,1] * gnb_bin_mean[,2],
         prod_knn = knn_bin_mean[,1] * knn_bin_mean[,2],
         prod_rf = rf_bin_mean[,1] * rf_bin_mean[,2],
         prod_xgb = xgb_bin_mean[,1] * xgb_bin_mean[,2]) %>%
  arrange(desc(mean[,2])) %>%
  print()

# Identify top 10 subgenre in terms of number of releases
top10_subg <- df_subg$subgenre[1:10]

# Check classifier performance
df_high_subg <- df_subg[1:10,]

# Check classifier performance against top 10 subgenres in order
gnb_order <- df_high_subg %>% arrange(desc(gnb_bin_mean[,1])) 
knn_order <- df_high_subg %>% arrange(desc(knn_bin_mean[,1]))
rf_order <- df_high_subg %>% arrange(desc(rf_bin_mean[,1]))
xgb_order <- df_high_subg %>% arrange(desc(xgb_bin_mean[,1]))
classifiers_order <- df_high_subg %>% arrange(desc(mean[,1])) 
  
gnb_recall_order <- sapply(top10_subg, function(x) 11 - grep(x,gnb_order$subgenre))
knn_recall_order <- sapply(top10_subg, function(x) 11 - grep(x,knn_order$subgenre))
rf_recall_order <- sapply(top10_subg, function(x) 11 - grep(x,rf_order$subgenre))
xgb_recall_order <- sapply(top10_subg, function(x) 11 - grep(x,xgb_order$subgenre))
classifiers_recall_order <- sapply(top10_subg, function(x) 11 - grep(x,classifiers_order$subgenre))


#### Check metrics vs precentage of high subgenres of the year the classifier was trained ####
df_subg_of_classifier_year <- 
  data_predict_subgenres %>%
  filter(release_year == 2009) %>%    #################### Change ME ###################
  filter(tag == "hiphop") %>% 
  #select(mp3_file_name:id | where(~is.numeric(.x) && sum(.x)>0)) %>%
  select(where(~!all(is.na(.x)))) %>%
  { map_dfr(.[,105:ncol(.)], 
            function(x) aggregate(list(gnb_bin_mean = .[["gnb_pred_bin_acc"]],
                                       knn_bin_mean = .[["knn_pred_bin_acc"]],
                                       rf_bin_mean = .[["rf_pred_bin_acc"]],
                                       xgb_bin_mean = .[["xgb_pred_bin_acc"]]), 
                                  list(group = x), 
                                  function(y) c(mean(y), length(y))),
            .id = "subgenre") } %>%
  mutate(mean = (gnb_bin_mean + knn_bin_mean + rf_bin_mean + xgb_bin_mean)/4) %>%
  mutate(prod = mean[,1] * mean[,2],
         prod_gnb = gnb_bin_mean[,1] * gnb_bin_mean[,2],
         prod_knn = knn_bin_mean[,1] * knn_bin_mean[,2],
         prod_rf = rf_bin_mean[,1] * rf_bin_mean[,2],
         prod_xgb = xgb_bin_mean[,1] * xgb_bin_mean[,2]) %>% print()

src_gnb_high_subg <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_gnb, decreasing = T),][1,1]
src_gnb_high_subg5 <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_gnb, decreasing = T),][1:5,1]
src_knn_high_subg <-df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_knn, decreasing = T),][1,1]
src_knn_high_subg5 <-df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_knn, decreasing = T),][1:5,1]
src_rf_high_subg <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_rf, decreasing = T),][1,1]
src_rf_high_subg5 <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_rf, decreasing = T),][1:5,1]
src_xgb_high_subg <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_xgb, decreasing = T),][1,1]
src_xgb_high_subg5 <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod_xgb, decreasing = T),][1:5,1]
src_classifiers_high_subg <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod, decreasing = T),][1,1]
src_classifiers_high_subg5 <- df_subg_of_classifier_year[order(df_subg_of_classifier_year$prod, decreasing = T),][1:5,1]

as.data.frame(tibble(
  src_gnb_high_subg = src_gnb_high_subg,
  src_gnb_high_subg5 = src_gnb_high_subg5,
  src_knn_high_subg = src_knn_high_subg,
  src_knn_high_subg5 = src_knn_high_subg5,
  src_rf_high_subg = src_rf_high_subg,
  src_rf_high_subg5 = src_rf_high_subg5,
  src_xgb_high_subg = src_xgb_high_subg,
  src_xgb_high_subg5 = src_xgb_high_subg5,
  src_classifiers_high_subg = src_classifiers_high_subg,
  src_classifiers_high_subg5 = src_classifiers_high_subg5
))


df_freq <-
  data_predict %>%
  filter(tag == "hiphop") %>%
  mutate(gnb_subg = ifelse(grepl(src_gnb_high_subg, genre),1,0),
         gnb_subg5 = ifelse(grepl(paste(paste("(",src_gnb_high_subg5, ")", sep=""), collapse = "|"), genre),1,0),
         knn_subg = ifelse(grepl(src_knn_high_subg, genre),1,0),
         knn_subg5 = ifelse(grepl(paste(paste("(",src_knn_high_subg5, ")", sep=""), collapse = "|"), genre),1,0),
         rf_subg = ifelse(grepl(src_rf_high_subg, genre),1,0),
         rf_subg5 = ifelse(grepl(paste(paste("(",src_rf_high_subg5, ")", sep=""), collapse = "|"), genre),1,0),
         xgb_subg = ifelse(grepl(src_xgb_high_subg, genre),1,0),
         xgb_subg5 = ifelse(grepl(paste(paste("(",src_xgb_high_subg5, ")", sep=""), collapse = "|"), genre),1,0),
         classifiers_subg = ifelse(grepl(src_classifiers_high_subg, genre),1,0),
         classifiers_subg5 = ifelse(grepl(paste(paste("(",src_classifiers_high_subg5, ")", sep=""), collapse = "|"), genre),1,0)
  ) %>%
  group_by(release_year) %>%
  summarize(gnb_mean = mean(gnb_pred_acc),
            knn_mean = mean(knn_pred_acc),
            rf_mean = mean(rf_pred_acc),
            xgb_mean = mean(xgb_pred_acc),
            classifiers_mean = mean(c(gnb_mean, knn_mean, rf_mean, xgb_mean)),
            gnb_freq = sum(gnb_subg)/n(),
            gnb_freq5 = sum(gnb_subg5)/n(),
            knn_freq = sum(knn_subg)/n(),
            knn_freq5 = sum(knn_subg5)/n(),
            rf_freq = sum(rf_subg)/n(),
            rf_freq5 = sum(rf_subg5)/n(),
            xgb_freq = sum(xgb_subg)/n(),
            xgb_freq5 = sum(xgb_subg5)/n(),
            classifiers_freq = sum(classifiers_subg)/n(),
            classifiers_freq5 = sum(classifiers_subg5)/n()
  ) %>%
  print()


df_metrics_freq <-
  tibble(
    release_year = data_predict_acc$release_year,
    gnb_bin_mean = data_predict_acc$gnb_bin_mean,
    knn_bin_mean = data_predict_acc$knn_bin_mean,
    rf_bin_mean = data_predict_acc$rf_bin_mean,
    xgb_bin_mean = data_predict_acc$xgb_bin_mean,
    classifiers_bin_mean = data_predict_acc$classifiers_bin_mean,
    gnb_recall = data_predict_recall$gnb_recall,
    knn_recall = data_predict_recall$knn_recall,
    rf_recall = data_predict_recall$rf_recall,
    xgb_recall = data_predict_recall$xgb_recall,
    classifiers_recall = data_predict_recall$classifiers_recall,
    gnb_freq = df_freq$gnb_freq,
    gnb_freq5 = df_freq$gnb_freq5,
    knn_freq = df_freq$knn_freq,
    knn_freq5 = df_freq$knn_freq5,
    rf_freq = df_freq$rf_freq,
    rf_freq5 = df_freq$rf_freq5,
    xgb_freq = df_freq$xgb_freq,
    xgb_freq5 = df_freq$xgb_freq5,
    classifiers_freq = df_freq$classifiers_freq,
    classifiers_freq5 = df_freq$classifiers_freq5
  )

summary(lm(gnb_bin_mean ~ gnb_freq, data = df_metrics_freq))
summary(lm(gnb_bin_mean ~ gnb_freq5, data = df_metrics_freq))
summary(lm(gnb_recall ~ gnb_freq, data = df_metrics_freq))
summary(lm(gnb_recall ~ gnb_freq5, data = df_metrics_freq))

summary(lm(knn_bin_mean ~ knn_freq, data = df_metrics_freq))
summary(lm(knn_bin_mean ~ knn_freq5, data = df_metrics_freq))
summary(lm(knn_recall ~ knn_freq, data = df_metrics_freq))
summary(lm(knn_recall ~ knn_freq5, data = df_metrics_freq))

summary(lm(rf_bin_mean ~ rf_freq, data = df_metrics_freq))
summary(lm(rf_bin_mean ~ rf_freq5, data = df_metrics_freq))
summary(lm(rf_recall ~ rf_freq, data = df_metrics_freq))
summary(lm(rf_recall ~ rf_freq5, data = df_metrics_freq))

summary(lm(xgb_bin_mean ~ xgb_freq, data = df_metrics_freq))
summary(lm(xgb_bin_mean ~ xgb_freq5, data = df_metrics_freq))
summary(lm(xgb_recall ~ xgb_freq, data = df_metrics_freq))
summary(lm(xgb_recall ~ xgb_freq5, data = df_metrics_freq))

summary(lm(classifiers_bin_mean ~ classifiers_freq, data = df_metrics_freq))
summary(lm(classifiers_bin_mean ~ classifiers_freq5, data = df_metrics_freq))
summary(lm(classifiers_recall ~ classifiers_freq, data = df_metrics_freq))
summary(lm(classifiers_recall ~ classifiers_freq5, data = df_metrics_freq))


#### Check tag/genre crossing ####
# T-test of accuracy of hip-hop crossing non-hip-hop songs against other songs
tibble_error <- list(p.value = c(NA), estimate = c(NA,NA))
possttest <- possibly(t.test, otherwise = tibble_error)

##### Table 5, Diff. Acc. #####
data_predict_subgenres %>%  # No tag and release year split
  filter(tag != "hiphop") %>%
  #filter(release_year > 2017) %>%
  #filter(tag != "folk") %>%
  mutate(rap = ifelse(grepl("(说唱)|(Hip-Hop)|(Trap)|(Rap)|(Hip Hop)", .$genre),1,0)) %>%
  select(rap, rf_pred_acc, knn_pred_acc, gnb_pred_acc, xgb_pred_acc) %>%
  #select(rap, xgb_pred_acc) %>%   # Change this #
  #filter(rap == 1) %>%
  group_by(rap) %>%
  nest() %>%
  #summarize(gnb_pred_acc = list(gnb_pred_acc)) %>%
  pivot_wider(names_from = rap, values_from = data, names_prefix = "hashiphop") %>% 
  mutate(n0 = length(unlist(hashiphop0)),
         value0 = possttest(unlist(hashiphop0), unlist(hashiphop1))$estimate[1],
         n1 = length(unlist(hashiphop1)),
         value1 = possttest(unlist(hashiphop0), unlist(hashiphop1))$estimate[2],
         diff = value1-value0,
         pval = round(possttest(unlist(hashiphop0), unlist(hashiphop1))$p.value, digits = 3),
         std = round(possttest(unlist(hashiphop0), unlist(hashiphop1))$stderr, digits = 3))

##### Table 5, Diff. FN #####
data_predict_subgenres %>%  # No tag and release year split
  filter(tag != "hiphop") %>%
  #filter(release_year > 2017) %>%
  #filter(tag != "folk") %>%
  mutate(rap = ifelse(grepl("(说唱)|(Hip-Hop)|(Trap)|(Rap)|(Hip Hop)", .$genre),1,0)) %>% 
  select(rap, rf_pred_bin_acc, knn_pred_bin_acc, gnb_pred_bin_acc, xgb_pred_bin_acc) %>%
  #select(rap, xgb_pred_bin_acc) %>%
  #filter(rap == 1) %>%
  group_by(rap) %>%
  nest() %>%
  #summarize(gnb_pred_acc = list(gnb_pred_bin_acc)) %>%
  pivot_wider(names_from = rap, values_from = data, names_prefix = "hashiphop") %>% 
  mutate(n0 = length(unlist(hashiphop0)),
         value0 = 1-possttest(unlist(hashiphop0), unlist(hashiphop1))$estimate[1],
         n1 = length(unlist(hashiphop1)),
         value1 = 1-possttest(unlist(hashiphop0), unlist(hashiphop1))$estimate[2],
         diff = value1-value0,
         pval = round(possttest(unlist(hashiphop0), unlist(hashiphop1))$p.value, digits = 3),
         std = round(possttest(unlist(hashiphop0), unlist(hashiphop1))$stderr, digits = 3))


