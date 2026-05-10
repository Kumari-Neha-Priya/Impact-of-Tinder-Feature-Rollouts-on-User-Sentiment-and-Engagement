---
title: "Impact of Tinder Feature Rollouts on User Sentiment and Engagement"
author: "Kumari Neha Priya, Si Thu Nyein Aung"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    toc_depth: 2
    number_sections: true
    theme: united
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Introduction
This analysis investigates how Tinder's feature rollouts affect user sentiment and engagement. We analyze Google and Apple play app store reviews and link them with a timeline of Tinder's major feature updates.

# Load Libraries and Data

```{r}
# Required packages
library(tidyverse)
library(lubridate)
library(sentimentr)
library(tm)
library(topicmodels)
library(tidytext)
library(scales)
library(ggplot2)
library(broom)
library(Metrics)
library(dplyr)
```
```{r}
# Set CRAN mirror first for reliable install
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Install xgboost (used in predictive modeling later)
install.packages("xgboost")

```

# Load Datasets

```{r}
# Load Tinder reviews from Google Play, Apple App Store, and feature release timeline
google <- read_csv("tinder_google_play_reviews.csv")
apple <- read_csv("tinder-apple-reviews.csv")
features <- read_csv("Timeline.csv")
```

# Inspect Datasets

```{r}
# Preview first few rows of each dataset
head(google)
head(apple)
head(features)

# Check column names for consistency and planning
colnames(google)
colnames(apple)
colnames(features)

```

# Data Cleaning and Preparation

## Google Reviews

```{r}
# Format dates and merge
google <- google %>%
  mutate(date = as.Date(.data$at)) %>%
  select(date, content, score, thumbsUpCount) %>%
  mutate(platform = "Google")

```


```{r}
# Check updated structure
head(google)
```

## Apple Reviews


```{r}
# Clean and rename columns for Apple data to match Google structure
apple <- apple %>% mutate(date = as.Date(date)) %>% select(date, review, rating) %>% rename(content = review, score = rating) %>% mutate(thumbsUpCount = NA, platform = "Apple")

```



```{r}
# Check Apple review structure after transformation
head(apple)

```

## Feature Timeline

```{r}
# Convert ReleaseDate to Date format
features <- features %>%
  mutate(ReleaseDate = as.Date(ReleaseDate, format = "%m/%d/%Y"))
```



```{r}
# Confirm structure
head(features)
colnames(features)
```

## Combine Google and Apple Reviews


```{r}
# Bind review datasets into a single DataFrame and drop missing content
reviews <- bind_rows(google, apple) %>% drop_na(content)

```



```{r}
# Assign unique ID to each review
reviews <- bind_rows(google, apple) %>%
  drop_na(content) %>%
  mutate(review_id = row_number())

```



```{r}
# Check combined dataset
head(reviews)
colnames(reviews)
```

# Match Reviews to Feature Rollouts

```{r}

# Custom function to assign the closest feature rollout date to each review
get_closest_feature <- function(review_date, rollout_dates) {
  # Check for features within 30 days after the review
  window_matches <- rollout_dates[rollout_dates - review_date <= 30 & rollout_dates - review_date >= 0]
  
  if (length(window_matches) > 0) {
    return(min(window_matches))  # upcoming feature within 30-day window
  }
  
  # Otherwise, return the latest past feature
  past_dates <- rollout_dates[rollout_dates <= review_date]
  if (length(past_dates) == 0) return(as.Date(NA))  # no match at all (pre-Tinder)
  return(max(past_dates))
}

# Apply matching function and create a sorted list of release dates
release_dates <- sort(unique(features$ReleaseDate))
reviews$feature_date <- as.Date(sapply(reviews$date, get_closest_feature, release_dates))

# Join feature metadata back into the review data using matched release date
reviews <- left_join(
  reviews,
  features %>% select(Feature, ReleaseDate, Type),
  by = c("feature_date" = "ReleaseDate")
)

```

# Define Pre/Post Periods Around Features

```{r}
# Define review period relative to feature rollout
window_days <- 30

reviews <- reviews %>%
  mutate(
    days_from_feature = as.numeric(date - feature_date),
    period = case_when(
      is.na(feature_date) ~ "NoFeature",  # Only for extremely early reviews
      days_from_feature >= -30 & days_from_feature < 0 ~ "Pre",  # Within 30 days before release
      days_from_feature >= 0 & days_from_feature <= 30 ~ "Post",  # Within 30 days after release
      TRUE ~ "Outside"
    )
  )

```

# Normalize Engagement Counts

```{r}
# Normalize thumbsUpCount based on how many features a review is matched to
reviews <- reviews %>%
  group_by(review_id) %>%
  mutate(feature_count = n(),
         normalized_thumbsUp = thumbsUpCount / feature_count) %>%
  ungroup()
```

# Create Unique Feature Identifiers

```{r}
# Create a unique identifier for each feature instance
reviews <- reviews %>%
  mutate(feature_id = paste0(Feature, "_", format(feature_date, "%Y%m%d")))

```



```{r}
# View updated data structure
head(reviews)
```

# Fill in Missing Values for Early or Unmatched Reviews

```{r}
# Set placeholders for reviews that don't match a known feature
reviews <- reviews %>%
  mutate(
    Feature = ifelse(is.na(Feature), "No Feature", Feature),
    Type = ifelse(is.na(Type), "None", Type),
    feature_id = ifelse(is.na(feature_date), paste0("NoFeature_", format(date, "%Y%m%d")), feature_id),
    feature_date = as.Date(feature_date)
  )


```



```{r}
# Set engagement metrics to 0 where thumbs-up is missing
reviews <- reviews %>%
  mutate(
    thumbsUpCount = ifelse(is.na(thumbsUpCount), 0, thumbsUpCount),
    normalized_thumbsUp = ifelse(is.na(normalized_thumbsUp), 0, normalized_thumbsUp)
  )
```

# Check for Missing Values

```{r}
# Summarize how many NA values exist per column
colSums(is.na(reviews))
```

# Sentiment Analysis Using sentimentr

```{r}
# Load necessary libraries for sentiment scoring
library(sentimentr)
library(data.table)

# Convert review dataset to data.table format for performance
setDT(reviews)

# Set size for batch processing to manage memory
chunk_size <- 10000
num_chunks <- ceiling(nrow(reviews) / chunk_size)

# Initialize progress message
cat("Starting sentiment analysis in", num_chunks, "chunks...\n")

# Create list to store results per chunk
sentiment_chunks <- vector("list", num_chunks)

# Loop through chunks of reviews and calculate sentiment
for (i in 1:num_chunks) {
  cat("Processing chunk", i, "of", num_chunks, "\n")
  
  start_row <- ((i - 1) * chunk_size + 1)
  end_row <- min(i * chunk_size, nrow(reviews))
  
  chunk <- reviews[start_row:end_row, ]
  
  # Safely get sentiment scores
  tryCatch({
    # Perform sentiment analysis using average sentiment per review
    chunk_sentiment <- sentiment_by(get_sentences(chunk$content))
    # Attach sentiment back to chunk
    chunk$sentiment <- chunk_sentiment$ave_sentiment
    sentiment_chunks[[i]] <- chunk
  }, error = function(e) {
    message("Error in chunk ", i, ": ", e$message)
    chunk$sentiment <- NA
    sentiment_chunks[[i]] <- chunk
  })
}

# Combine all processed chunks back into one dataset
reviews <- rbindlist(sentiment_chunks)
```

## Classify Sentiment into Labels

```{r}
# Convert numeric sentiment scores into categories
reviews <- reviews %>%
  mutate(
    sentiment_label = case_when(
      sentiment > 0.1 ~ "Positive",  # Above threshold = positive
      sentiment < -0.1 ~ "Negative", # Below threshold = negative
      TRUE ~ "Neutral"               # Otherwise neutral
    )
  )

```

## Preview and Save the Processed Dataset

```{r}
# Preview result
head(reviews)
```



```{r}
# Save the enriched dataset for downstream analysis (classification, modeling, etc.)
write_csv(reviews, "reviews final to be used for further steps new.csv")

```

# Reload Raw Reviews with Reply Content

```{r}
# Load original Google reviews including developer replies
google_raw <- read_csv("tinder_google_play_reviews.csv") %>%
  mutate(date = as.Date(at)) %>%
  select(date, content, replyContent)

# Load original Apple reviews with reply metadata
apple_raw <- read_csv("tinder-apple-reviews.csv") %>%
  mutate(date = as.Date(date)) %>%
  select(date, review, developerResponse, isEdited) %>%
  rename(content = review)


```



```{r}
# Preview loaded data
head(google_raw)
head(apple_raw)
```

# Join Developer Response Fields

```{r}
# Add Google developer replies to main review dataset
reviews <- reviews %>%
  left_join(google_raw, by = c("date", "content"))  # adds replyContent

# Add Apple developer responses
reviews <- reviews %>%
  left_join(apple_raw, by = c("date", "content"))  # adds developerResponse

```



```{r}
# Verify joined structure
head(reviews)
```

# Predictive Analysis

## Logistic Regression

```{r}
# Load ML-related libraries
library(tidyverse)
library(ROSE)
library(pROC)
library(caret)

# --- FEATURE ENGINEERING ---
# Add useful predictors
reviews <- reviews %>%
  mutate(
    is_positive = ifelse(sentiment_label == "Positive", 1, 0),
    Paid = ifelse(Type == "Paid", 1, 0),
    review_length = str_count(content, "\\w+"),
    has_thumbsup = ifelse(thumbsUpCount > 0, 1, 0),
    has_dev_response = ifelse(!is.na(replyContent) | !is.na(developerResponse), 1, 0),
    isEdited = ifelse(is.na(isEdited), FALSE, isEdited)
  )

# Split Data into Pre/Post for Model Training
# Keep only Pre and Post data for modeling
model_data <- reviews %>%
  filter(!is.na(Type) & period %in% c("Pre", "Post")) %>%
  select(is_positive, Paid, platform, normalized_thumbsUp, review_length, 
         has_thumbsup, has_dev_response, isEdited, period)

# Training = Pre period; Testing = Post period
train_data <- model_data %>% filter(period == "Pre") %>% select(-period)
test_data  <- model_data %>% filter(period == "Post") %>% select(-period)

# Convert factors and booleans appropriately
train_data <- train_data %>%
  mutate(
    platform = factor(platform),
    isEdited = as.integer(isEdited)  # TRUE/FALSE → 1/0
  )

test_data <- test_data %>%
  mutate(
    platform = factor(platform),
    isEdited = as.integer(isEdited)
  )

# --- BALANCE TRAINING SET ---
# Balance binary classes using ROSE
set.seed(123)
balanced_train <- ROSE(is_positive ~ ., data = train_data, seed = 123)$data

# --- MODEL ---
# Logistic Regression Modeling
# Fit logistic regression on balanced training data
logit_model <- glm(is_positive ~ ., data = balanced_train, family = "binomial")
# View model summary
summary(logit_model)

# --- PREDICT ---
# Predict probabilities (likelihood of "Positive")
train_data$pred_prob <- predict(logit_model, newdata = train_data, type = "response")
test_data$pred_prob  <- predict(logit_model, newdata = test_data, type = "response")

# --- OPTIMAL THRESHOLD ---
# Compute ROC curve on training predictions
roc_obj <- roc(train_data$is_positive, train_data$pred_prob)
# Determine optimal threshold from ROC
opt_thresh <- coords(roc_obj, "best", ret = "threshold")
# Manually define threshold
custom_thresh <- 0.50

# Predict using custom threshold
train_data$pred_class <- ifelse(train_data$pred_prob >= custom_thresh, 1, 0)
test_data$pred_class <- ifelse(test_data$pred_prob >= custom_thresh, 1, 0)

# --- METRICS ---
# Calculate AUC scores
train_auc <- auc(train_data$is_positive, train_data$pred_prob)
test_auc  <- auc(test_data$is_positive, test_data$pred_prob)

# Print results
cat("\n Train AUC:", round(train_auc, 3), "\n")
print(confusionMatrix(factor(train_data$pred_class), factor(train_data$is_positive), positive = "1"))

cat("\n Test AUC:", round(test_auc, 3), "\n")
print(confusionMatrix(factor(test_data$pred_class), factor(test_data$is_positive), positive = "1"))


```



```{r}
# Convert the response variable to factor for classification
train_data$is_positive <- as.factor(train_data$is_positive)
test_data$is_positive  <- as.factor(test_data$is_positive)

```

## Random Forest

```{r}
# Load library and set seed for reproducibility
library(randomForest)
set.seed(123)

# Convert categorical variables
train_data$platform <- as.factor(train_data$platform)
test_data$platform <- as.factor(test_data$platform)

# Train Random Forest model with 100 trees
rf_model <- randomForest(
  is_positive ~ ., 
  data = train_data, 
  ntree = 100,
  importance = TRUE
)

# Predict probabilities
train_data$rf_prob <- predict(rf_model, train_data, type = "prob")[,2]
test_data$rf_prob <- predict(rf_model, test_data, type = "prob")[,2]

# Classification with threshold 0.50
train_data$rf_pred <- ifelse(train_data$rf_prob >= 0.50, 1, 0)
test_data$rf_pred <- ifelse(test_data$rf_prob >= 0.50, 1, 0)

# Evaluate model using AUC and confusion matrix
library(pROC)
train_auc_rf <- auc(train_data$is_positive, train_data$rf_prob)
test_auc_rf <- auc(test_data$is_positive, test_data$rf_prob)

cat("\nTrain AUC:", round(train_auc_rf, 3), "\n")
cat("Test AUC:", round(test_auc_rf, 3), "\n")

confusionMatrix(factor(test_data$rf_pred), factor(test_data$is_positive), positive = "1")

```

# XGBoost

```{r}
# Load libraries
library(xgboost)
library(Matrix)
library(pROC)
library(caret)

# Create new training and test sets for XGBoost
xgb_train <- model_data %>% 
  filter(period == "Pre") %>%
  select(-period)

xgb_test <- model_data %>% 
  filter(period == "Post") %>%
  select(-period)

# Convert factors to numeric matrix
train_x <- model.matrix(is_positive ~ . - 1, data = xgb_train)
test_x  <- model.matrix(is_positive ~ . - 1, data = xgb_test)

# Extract target labels
train_y <- as.numeric(as.character(xgb_train$is_positive))
test_y  <- as.numeric(as.character(xgb_test$is_positive))


# -------------------------------
# Train XGBoost model
# -------------------------------
set.seed(123)
xgb_model <- xgboost(
  data = train_x,
  label = train_y,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "auc",
  verbose = 0
)

# -------------------------------
# 3. Predict
# -------------------------------
train_pred_prob <- predict(xgb_model, train_x)
test_pred_prob  <- predict(xgb_model, test_x)

# Classification at 0.50 threshold
train_pred <- ifelse(train_pred_prob >= 0.50, 1, 0)
test_pred  <- ifelse(test_pred_prob >= 0.50, 1, 0)

# -------------------------------
# 4. Evaluate
# -------------------------------
# Calculate AUC scores
train_auc <- auc(train_y, train_pred_prob)
test_auc  <- auc(test_y, test_pred_prob)

cat("\nTrain AUC:", round(train_auc, 3), "\n")
cat("Test AUC:", round(test_auc, 3), "\n")

# Display confusion matrix
confusionMatrix(factor(test_pred), factor(test_y), positive = "1")

```

## Cross-Validated XGBoost via caret (Finalized Predictive Model)

```{r}
# Load libraries
library(caret)
library(xgboost)
library(Matrix)
library(pROC)

# 1. Recreate training and test sets
xgb_train <- model_data %>% 
  filter(period == "Pre") %>%
  select(-period)
xgb_test  <- model_data %>% 
  filter(period == "Post") %>%
  select(-period)

# 2. Prepare matrices and labels
# Convert is_positive to factor yes/no for caret
xgb_train$is_positive <- factor(ifelse(xgb_train$is_positive == 1, "yes", "no"))
xgb_test$is_positive  <- factor(ifelse(xgb_test$is_positive  == 1, "yes", "no"))

train_x <- model.matrix(is_positive ~ . - 1, data = xgb_train)
test_x  <- model.matrix(is_positive ~ . - 1, data = xgb_test)

train_y <- xgb_train$is_positive
test_y  <- xgb_test$is_positive

# 3. Set up 5-fold CV control
cv_ctrl <- trainControl(
  method              = "cv",
  number              = 5,
  summaryFunction     = twoClassSummary,
  classProbs          = TRUE,
  verboseIter         = TRUE,
  savePredictions     = "final",
  allowParallel       = TRUE
)

# 4. Train with caret (using default xgbTree grid)
set.seed(123)
xgb_caret <- train(
  x            = train_x,
  y            = train_y,
  method       = "xgbTree",
  metric       = "ROC",       # optimize AUC
  trControl    = cv_ctrl,
  tuneLength   = 5            # try 5 levels of each tuning param
)

# View the best tuning parameters and CV AUC
print(xgb_caret)
plot(xgb_caret)

# 5. Predict on holdout “Post” set
test_probs <- predict(xgb_caret, test_x, type = "prob")[, "yes"]
test_pred  <- ifelse(test_probs >= 0.5, "yes", "no")

# 6. Evaluate
test_auc <- roc(response = test_y, predictor = test_probs)
cat("Test AUC:", round(test_auc$auc, 3), "\n")

confusionMatrix(
  factor(test_pred, levels = c("no","yes")),
  test_y,
  positive = "yes"
)


```

## Feature Importance for  Cross Validated XGBoost (Finalized Predictive Model)

```{r}
library(xgboost)

# Plot feature importance
importance_matrix <- xgb.importance(model = xgb_model, feature_names = colnames(train_x))

# Plot top features with title
xgb.plot.importance(importance_matrix, top_n = 10, rel_to_first = TRUE, 
                    xlab = "Relative Importance", col="steelblue", 
                    main = "XGBoost Feature Importance")


```

# Causal Analysis (DiD)

## Feature-Specific DiD Model: Engagement


```{r}
# Prepare data for engagement DiD (thumbs up as proxy)
engage_data <- reviews %>%
  filter(!is.na(Type), period %in% c("Pre", "Post")) %>%
  mutate(
    Post = ifelse(period == "Post", 1, 0),
    Paid = ifelse(Type == "Paid", 1, 0),
    thumbsUpCount = ifelse(is.na(thumbsUpCount), 0, thumbsUpCount)
  )

# Run DiD regression on thumbs up
# Feature-specific engagement model
engage_did_feature <- lm(
  thumbsUpCount ~ Post * Paid * Feature + platform,
  data = engage_data
)
summary(engage_did_feature)

```

## Visualize Top Post-Rollout Feature Effects


```{r}
# Extract and format coefficients
coefs <- summary(engage_did_feature)$coefficients
engage_effects <- as.data.frame(coefs[grepl("Post:Feature", rownames(coefs)), ])
engage_effects$Feature <- gsub("Post:Feature", "", rownames(engage_effects))
engage_effects <- engage_effects %>%
  rename(Estimate = Estimate) %>%
  arrange(desc(Estimate))

# Top 10 positively engaging features
top_engage <- engage_effects %>% top_n(10, Estimate)

# Plot
ggplot(top_engage, aes(x = reorder(Feature, Estimate), y = Estimate)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top Features by Post-Rollout Engagement Impact",
    x = "Feature",
    y = "Effect on Thumbs-Up Count (DiD)"
  ) +
  theme_minimal()

```

## Feature-Specific DiD Model: Sentiment

```{r}
# Base dataset
did_data <- reviews %>%
  filter(!is.na(Type) & period %in% c("Pre", "Post")) %>%
  mutate(
    Post = ifelse(period == "Post", 1, 0),
    Paid = ifelse(Type == "Paid", 1, 0),
    normalized_thumbsUp = ifelse(is.na(normalized_thumbsUp), 0, normalized_thumbsUp)
  )

# DiD Regression
did_model <- lm(
  sentiment ~ Post * Paid * Feature + platform + normalized_thumbsUp,
  data = did_data
)
# View output: Post:Paid:Feature interactions
summary(did_model)


```

## Visualize Top Post-Rollout Feature Effects


```{r}
# Plot top features by sentiment improvement after release
library(ggplot2)

# Use interaction effects only (e.g., Post:Feature coefficients)
coefs <- summary(did_model)$coefficients
interaction_effects <- as.data.frame(coefs[grepl("Post:Feature", rownames(coefs)), ])
interaction_effects$Feature <- gsub("Post:Feature", "", rownames(interaction_effects))
interaction_effects <- interaction_effects %>%
  rename(Estimate = Estimate) %>%
  arrange(desc(Estimate))

# Plot top 5 sentiment improvements
top_feature_effects <- interaction_effects %>% top_n(5, Estimate)

ggplot(top_feature_effects, aes(x = reorder(Feature, Estimate), y = Estimate)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Top Features by Post-Rollout Sentiment Change (DiD)",
    x = "Feature",
    y = "Effect on Sentiment (VADER)"
  ) +
  theme_minimal()
```

## Review Volume and Avg Thumbs Up Before vs After by Feature Type

```{r}
# -- Review volume by period and Type (Free vs Paid) --
review_volume <- reviews %>%
  filter(!is.na(Type), period %in% c("Pre", "Post")) %>%
  mutate(period = factor(period, levels = c("Pre", "Post"))) %>%  # Set order
  group_by(period, Type) %>%
  summarise(review_count = n(), .groups = "drop")

# Plot
ggplot(review_volume, aes(x = period, y = review_count, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Review Volume Before vs After by Feature Type", y = "Number of Reviews") +
  theme_minimal()

# -- Average Thumbs Up by period and Type --
avg_thumbs <- reviews %>%
  filter(!is.na(Type), period %in% c("Pre", "Post")) %>%
  mutate(period = factor(period, levels = c("Pre", "Post"))) %>%  # Set order
  group_by(period, Type) %>%
  summarise(mean_thumbs = mean(thumbsUpCount, na.rm = TRUE), .groups = "drop")

# Plot
ggplot(avg_thumbs, aes(x = period, y = mean_thumbs, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Avg Thumbs Up Before vs After by Feature Type", y = "Avg Thumbs Up Count") +
  theme_minimal()

```


# Topic Modeling (LDA)

```{r}
library(topicmodels)
library(tidytext)
library(tm)
library(dplyr)

# STEP 1: Sample a subset of reviews and prepare data
sampled_reviews <- reviews %>% 
  filter(!is.na(content)) %>%
  sample_n(5000)

# STEP 2: Create text corpus and clean text
corpus <- Corpus(VectorSource(sampled_reviews$content)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, c(stopwords("english"), "tinder", "app", "feature")) %>% # remove common noise
  tm_map(stripWhitespace)

# STEP 3: Create document-term matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)

# Remove empty rows
row_totals <- apply(dtm, 1, sum)
dtm <- dtm[row_totals > 0, ]


# STEP 4: Fit LDA
lda_model <- LDA(dtm, k = 3, control = list(seed = 1234))
topics <- tidy(lda_model, matrix = "beta")

# STEP 5: Extract top terms for each topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  arrange(topic, -beta)

# Visualize top terms per topic
library(ggplot2)
top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top Terms per Topic", x = "Beta", y = NULL) +
  theme_minimal()

```

# Post Modeling Visualizations

## Sentiment by feature

```{r}
# Aggregate average and median sentiment for each feature
reviews %>%
  group_by(Feature) %>%
  summarise(avg_sentiment = mean(sentiment, na.rm = TRUE),
            median_sentiment = median(sentiment, na.rm = TRUE),
            review_count = n()) %>%
  arrange(desc(avg_sentiment))

```

## Visualize Top 10 Features by Sentiment

```{r}
sentiment_by_feature <- reviews %>%
  filter(!is.na(Feature)) %>%
  group_by(Feature) %>%
  summarise(
    avg_sentiment = mean(sentiment, na.rm = TRUE),
    median_sentiment = median(sentiment, na.rm = TRUE),
    review_count = n()
  ) %>%
  arrange(desc(avg_sentiment))

# Plot top 10 features by average sentiment
top_features <- sentiment_by_feature %>% top_n(10, avg_sentiment)

ggplot(top_features, aes(x = reorder(Feature, avg_sentiment), y = avg_sentiment)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Features by Avg Sentiment",
    x = "Feature",
    y = "Avg Sentiment"
  ) +
  theme_minimal()

```

## Engagement by feature (Thumbs-Up Count)

```{r}
reviews %>%
  group_by(Feature) %>%
  summarise(avg_thumbs = mean(thumbsUpCount, na.rm = TRUE),
            review_volume = n()) %>%
  arrange(desc(avg_thumbs))

```

## Top 10 Features by Avg Thumbs-Up

```{r}
engagement_by_feature <- reviews %>%
  filter(!is.na(Feature)) %>%
  group_by(Feature) %>%
  summarise(
    avg_thumbs = mean(thumbsUpCount, na.rm = TRUE),
    review_count = n()
  ) %>%
  arrange(desc(avg_thumbs))

# Plot top 10 engaging features
top_engaging <- engagement_by_feature %>% top_n(10, avg_thumbs)

ggplot(top_engaging, aes(x = reorder(Feature, avg_thumbs), y = avg_thumbs)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top 10 Features by Avg Thumbs-Up",
    x = "Feature",
    y = "Avg Thumbs-Up Count"
  ) +
  theme_minimal()

```

## Top 10 Most Reviewed Feature

```{r}
# Check most discussed features (across all time)
top_features_volume <- reviews %>%
  filter(!is.na(Feature)) %>%
  group_by(Feature) %>%
  summarise(review_count = n(), .groups = "drop") %>%
  arrange(desc(review_count)) %>%
  top_n(10, review_count)

# Plot
ggplot(top_features_volume, aes(x = reorder(Feature, review_count), y = review_count)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(
    title = "Top 10 Most Reviewed Features",
    x = "Feature",
    y = "Number of Reviews"
  ) +
  theme_minimal()


```

## Sentiment Extremes Over Time

```{r}

# Track % of extreme sentiments (positive or negative) by month
variation_ts <- reviews %>%
  mutate(
    extreme = sentiment_label %in% c("Positive","Negative"),
    month   = floor_date(date, "month")
  ) %>%
  group_by(month) %>%
  summarise(
    pct_extreme = mean(extreme, na.rm = TRUE),
    .groups = "drop"
  )

# Plot
ggplot(variation_ts, aes(x = month, y = pct_extreme)) +
  geom_line(size = 1) +
  geom_point() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Variation in Sentiment (Extremes) Over Time",
    subtitle = "% of Reviews That Are Positive or Negative",
    x = "Month",
    y = "Percent Extreme"
  ) +
  theme_minimal()

```

## Focus on Key Features for Case Study


```{r}
# Focus on 3–5 Key Features for Case Studies
# Define key features for focused visual analysis
key_features <- c("Boost", "Passport", "Likes You", "Superlike")

# Filter review data for key features and pre/post periods
did_key <- reviews %>%
  filter(Feature %in% key_features, period %in% c("Pre","Post")) %>%
  mutate(period = factor(period, levels=c("Pre","Post")))

# Plot sentiment change for selected features
ggplot(did_key, aes(x = period, y = sentiment, color = Feature, group = Feature)) +
  stat_summary(fun = mean, geom = "line", size = 1.2) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(
    title = "Sentiment Change for Key Feature Rollouts",
    subtitle = paste(key_features, collapse=", "),
    x = "Period",
    y = "Avg Sentiment"
  ) +
  theme_minimal()
```

## Monthly Sentiment Proportions with Feature Rollouts

```{r sentiment_plot, fig.width=18, fig.height=6}
library(ggplot2)
library(dplyr)
library(lubridate)
library(scales)
library(stringr)

# Step 1: Monthly sentiment proportions
sentiment_monthly <- reviews %>%
  filter(!is.na(sentiment_label)) %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month, sentiment_label) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(month) %>%
  mutate(prop = n / sum(n))

# Step 2: Combine overlapping feature labels by date
features_to_label <- features %>%
  filter(!is.na(Feature)) %>%
  group_by(ReleaseDate) %>%
  summarise(Feature = paste(unique(Feature), collapse = ", ")) %>%
  ungroup() %>%
  arrange(ReleaseDate) %>%
  mutate(
    Feature = str_wrap(Feature, width = 15),
    y_pos = rep(c(1.02, 1.08, 1.14), length.out = n())  # 3 staggered levels
  )

# Step 3: Plot with improved label formatting
ggplot(sentiment_monthly, aes(x = month, y = prop, color = sentiment_label)) +
  geom_line(size = 1) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.2)) +
  scale_color_manual(values = c("Negative" = "red", "Neutral" = "green", "Positive" = "blue")) +
  labs(
    title = "Monthly Sentiment Proportions with Feature Rollouts",
    y = "Proportion of Reviews",
    x = "Month",
    color = "Sentiment"
  ) +
  theme_minimal(base_size = 14) +
  geom_vline(data = features_to_label, aes(xintercept = as.numeric(ReleaseDate)), 
             linetype = "dashed", color = "gray60") +
  geom_text(data = features_to_label, 
            aes(x = ReleaseDate, y = y_pos, label = Feature), 
            angle = 90, vjust = 0, size = 3.2, color = "black")

```

## Rating vs. VADER Sentiment Agreement

```{r}
library(dplyr)
library(ggplot2)
library(reshape2)
library(scales)

# 1. Map rating to sentiment
reviews <- reviews %>%
  mutate(
    rating_sentiment = case_when(
      score >= 4 ~ "positive",
      score <= 2 ~ "negative",
      TRUE       ~ "neutral"
    )
  )

# 2. Confusion matrix (Rating vs VADER)
conf_matrix <- table(reviews$rating_sentiment, reviews$sentiment_label)

# Normalize by row to get proportions
conf_prop <- prop.table(conf_matrix, margin = 1)  # row-wise proportions
conf_df <- as.data.frame(conf_prop)
colnames(conf_df) <- c("RatingSentiment", "VaderSentiment", "Proportion")

# 3. Plot normalized heatmap
ggplot(conf_df, aes(x = VaderSentiment, y = RatingSentiment, fill = Proportion)) +
  geom_tile(color = "gray90", linewidth = 0.5) +
  geom_text(aes(label = percent(Proportion, accuracy = 1)), size = 5) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.33) +
  labs(
    title = "Rating-Based vs VADER Sentiment (Proportions)",
    x = "VADER Sentiment",
    y = "Rating-Based Sentiment",
    fill = "Proportion"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 12),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )


```

##  Compare Sentiment by Platform

```{r}
library(dplyr)
library(ggplot2)
library(scales)

# 1. Group by platform and sentiment, count reviews
prop_df <- reviews %>%
  group_by(platform, sentiment_label) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(platform) %>%
  mutate(proportion = count / sum(count))

# 2. Custom color palette: Apple gray (#A2AAAD), Android green (#3DDC84)
custom_colors <- c("Apple" = "#A2AAAD", "Google" = "#3DDC84")

# 3. Plot
ggplot(prop_df, aes(x = sentiment_label, y = proportion, fill = platform)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "Standardized Sentiment Distribution by Platform",
    x = "Sentiment",
    y = "Proportion of Reviews",
    fill = "Platform"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )


```

