#################################################################################################

# The aim of this script is to create a dataset with the following information:
#   - Name of the article
#   - Content of the article
#   - Category of the article

# This is a rough outline that is used to covert a txt file into the form that we need to do the
# raw text analysis. This is to be editied to work with out specific dataset although it should
# follow the same structure mentioned above [originally written based off the BBC dataset]

#################################################################################################

# Installs
install.packages("readtext", dependencies=T)
install.packages("magrittr") # package installations are only needed the first time you use it
install.packages("dplyr")    # alternative installation of the %>%


# Imports
library(readtext)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%


# Cleaning environment data
rm(list = ls())

# Working directory
setwd('D:/Graduate School/text-classifier')

# Path definition of the news archives
path <- 'D:/Graduate School/text-classifier/data/gdelt'

# List with the 5 categories
list_categories <- list.files(path=path)

# Save to dataset the number of files in each category folder
summary_categories <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(summary_categories) <- c('Category', 'Number_of_Files')

for (category in list_categories){
  category_path <- paste(path, category, sep='/')
  n_files <- length(list.files(path=category_path))

  summary_categories = rbind(summary_categories, data.frame('Category'=category, 'Number_of_Files'=n_files))
}

summary_categories

# Read every folder and create the final dataframe
df_final <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(df_final) <- c('doc_id', 'text', 'category')

for(category in list_categories){
  category_path <- paste(path, category, sep='/')

  df <- readtext(category_path)
  df["category"] = category

  df_final = rbind(df_final, df)
}

colnames(df_final) <- c('File_Name', 'Content', 'Category')

df_final <-
  df_final %>%
    mutate(Complete_Filename = paste(File_Name, Category, sep='-'))

# Save dataset: .rda
# TODO: Rename DATASET/change DATASET location
save(df_final, file='data/Dataset_india.rda')

# Load dataset
# TODO: if dataset changes above change it here
load(file='data/Dataset_india.rda')

# Write csv to import to python
# TODO: change name of final saved dataset 
write.csv2(df_final,fileEncoding = 'utf8', "data/News_India_dataset.csv", row.names = FALSE)