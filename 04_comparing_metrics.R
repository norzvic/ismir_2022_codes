#### Preparing working environment ####
# Set current folder as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, ggpubr, lubridate)

# Load data
data <- readRDS('song_data.rds')

#### Preliminary Checks ####
# Check genre composition over time
data_genres_composition <- 
  data %>%
  filter(release_year %in% c("2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019")) %>%
  mutate(release_year = year(release_date)) %>%
  group_by(release_year, tag) %>%
  summarize(n=n()) %>%
  spread(tag, n) %>%
  mutate(sum = folk + hiphop + pop + rock,
         folk_prop = folk/sum,
         hiphop_prop = hiphop/sum,
         pop_prop = pop/sum,
         rock_prop = rock/sum) %>%
  print()

data_genres_composition_GTZAN <- 
  data %>%
  filter(release_year %in% c("2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019")) %>%
  filter(tag != "folk") %>%
  mutate(release_year = year(release_date)) %>%
  group_by(release_year, tag) %>%
  summarize(n=n()) %>%
  spread(tag, n) %>%
  mutate(sum_GTZAN = hiphop + pop + rock,
         hiphop_prop_GTZAN = hiphop/sum_GTZAN,
         pop_prop_GTZAN = pop/sum_GTZAN,
         rock_prop_GTZAN = rock/sum_GTZAN) %>%
  select(1,5:8) %>%
  print()

# Load metrics data: Need to run the first three R scripts
filenames <- list.files(pattern = "_metrics_")
read_list <- lapply(filenames, readRDS)
names(read_list) <- gsub(".rds","",filenames)
merge <- map_dfr(read_list, bind_rows,.id = "src")  # Merge into one 

# Indexing classifier & data source
merge <- merge %>%
  mutate(classifier = gsub("(^[a-z]{2,})_.*$", "\\1", src),
         data = gsub("^.*_metrics_(.*)$", "\\1", src))

# Combining with proportion data
merge <- merge %>%
  right_join(data_genres_composition, by = "release_year") %>%
  right_join(data_genres_composition_GTZAN, by = "release_year")
View(merge)

# Check polynomial regression
lm_polynomial <- 
  merge %>%
  #filter(release_year != 2009) %>% 
  filter(grepl('mean_pred_bin_acc', metrics) | grepl('recall', metrics)) %>% 
  filter(!grepl('trap', metrics)) %>%
  group_by(src, metrics) %>%
  #summarize(mean = mean(years))
  summarize(est_year = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[2,1],
            sd_year = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[2,2],
            pval_year = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[2,4],
            est_year2 = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[3,1],
            sd_year2 = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[3,2],
            pval_year2 = summary(lm(values ~ release_year + I(release_year^2)))$coefficients[3,4],
            r2 = summary(lm(values ~ release_year + I(release_year^2)))$r.squared) %>% 
  arrange(metrics) %>%
  select(src, metrics, pval_year,pval_year2) %>%
  filter(grepl('acc',metrics)) 

lm_polynomial$pval_year2 <0.05


# Visualize: Mean Prediction Hip-Hop Binary & Recall
plot_accuracy <-
merge %>%
  filter(grepl("mean_pred_bin", metrics) == TRUE) %>%
  filter(data %in% c("2009","2018","GTZAN")) %>%
  ggplot(aes(x = release_year, y = values, shape = metrics)) +
  #geom_point(size = 1.5) +
  geom_line(aes(linetype = data, color = data)) +
  geom_vline(xintercept = 2017, linetype = "dashed") +
  #scale_linetype_manual(values=c("solid", "dotted", "solid", "dotted", "solid", "dotted")) +
  facet_grid(~ factor(classifier, 
                      levels = c("gnb","knn","rf",'xgb',"classifiers"), 
                      labels = c("GNB","KNN","RF",'XGB',"Average"))) +
  geom_rect(aes(xmin = 2017, xmax = Inf, ymin = -Inf, ymax = Inf), fill = "grey45", alpha = 0.01) +
  scale_x_continuous(breaks = seq(2010,2018,4)) +
  scale_color_manual(name = "Cohort", values = c("grey2","grey10","grey15")) +
  scale_linetype_manual(name = "Cohort", values = c("solid","longdash","dotted")) +
  theme(legend.position = "bottom") +
  labs(
    #title = "Accuracy Hip-Hop",
    x = "",
    y = "Accuracy")

plot_recall <-
merge %>%
  filter(grepl("recall", metrics) == TRUE) %>%
  filter(grepl("trap", metrics) == FALSE) %>%
  filter(data %in% c("2009","2018","GTZAN")) %>%
  ggplot(aes(x = release_year, y = values, shape = metrics)) +
  #geom_point(size = 1.5) +
  geom_line(aes(linetype = data, color = data)) +
  geom_vline(xintercept = 2017, linetype = "dashed") +
  #scale_linetype_manual(values=c("solid", "dotted", "solid", "dotted", "solid", "dotted")) +
  facet_grid(~ factor(classifier, 
                      levels = c("gnb","knn","rf",'xgb',"classifiers"), 
                      labels = c("GNB","KNN","RF",'XGB',"Average"))) +  
  geom_rect(aes(xmin = 2017, xmax = Inf, ymin = -Inf, ymax = Inf), fill = "grey45", alpha = 0.01) +
  scale_x_continuous(breaks = seq(2010,2018,4)) +
  scale_color_manual(name = "Cohort", values = c("grey2","grey10","grey15")) +
  scale_linetype_manual(name = "Cohort", values = c("solid","longdash","dotted")) +
  theme(legend.position = "bottom") +
  labs(
    #title = "Accuracy Hip-Hop",
    x = "Release Year",
    y = "Recall")

ggarrange(plot_accuracy,plot_recall,nrow = 2, common.legend = T, legend = "bottom")
  
