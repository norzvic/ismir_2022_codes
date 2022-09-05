#### Preparing Data ####
# Set working directory
setwd("~/Dropbox/academia/Papers/Work in Progress/2022 Classfier inaccuracy and genre evolution/github_upload/")

# Load packages
library(tidyverse)

# Clean data
load("../Analysis/data.RData")
data <- data %>%
  select(song_id, tag, release_date, release_month, release_year, listen_times, 
         chroma_stft_norm:mfcc20_norm)

saveRDS(data, file='song_data.rds')

