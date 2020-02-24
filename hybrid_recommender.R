#####################################################################################################
#####################################################################################################
# PH125.9x - Data Science : Capstone
# Movielens Recommendation Assignment
# C Yáñez Santibáñez
#####################################################################################################
#####################################################################################################
# This file contains code used to generate movie rating predictions from the movielens dataset.
#This file is structured in the following sections:
# 1. Code to load all require libraries
# 2. Machine learning model and evaluation functions
#### 2.1 RMSE calculator
#### 2.2 TidyUp movielens data set
#### 2.3 Split data into train and test set
#### 2.4 Create User Vectors
#### 2.5 Classify (Cluster) users
#### 2.6 Calculate biases for general model
#### 2.7 Calculate biases for cluster-based model
#### 2.8 Predict Rating 
#### 2.9 Load data from Movielens source (code provided in course materials)
# 3. Code to run prediction with optimal parameters (commented)

#####################################################################################################
#####################################################################################################
# 1. Code to load all require libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
if(!require(ClusterR)) install.packages("ClusterR", repos = "http://cran.us.r-project.org")

#####################################################################################################
#####################################################################################################
# 2. Machine learning model and evaluation functions


#### 2.1 RMSE calculator (self explanatory)
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### 2.2 TidyUp movielens data set
Tidy_Up <- function(movielens, title = TRUE, time_stamp = TRUE) {
#  if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  
  # creating copy, just in case original data is needed for additional transformations
  temp <- movielens

  # Get release year out of title
  if (title == TRUE) {
    temp <- temp %>%
      mutate(title_old = title, title = str_remove(title_old, " \\(([1-2][0-9][0-9][0-9])\\)"), release_year = str_extract(title_old, "[1-2][0-9][0-9][0-9]")) %>%
      select(-title_old)
  }

  # Get date out of timestamp
  if (title == TRUE) {
    temp <- temp %>% mutate(rating_date = as.Date(as.POSIXct(.$timestamp, origin = "1970-01-01")), rating_year = as.numeric(format(rating_date, "%Y")))
  }

  # add sequential number - easier for matching operations later
  temp$row_id <- seq.int(nrow(temp))

  # return data
  temp
}

#### 2.3 Split data into train and test set.
Train_Test <- function(movielens) {
  
  #Parameters:
  #movielens : movielens dataset  
  
  if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  # REUSING code provided in course materials.

  # set.seed(seed, sample.kind="Rounding")
  # if using R 3.5 or earlier, use `set.seed(1)` instead
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  train <- movielens[-test_index, ]
  temp <- movielens[test_index, ]

  # Make sure userId and movieId in validation set are also in train set
  test <- temp %>%
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")

  # Add rows removed from validation set back into train set
  removed <- anti_join(temp, test)
  train <- rbind(train, removed)

  # collate into output structure
  output <- vector(mode = "list", length = 0)
  output$train <- train
  output$test <- test

  output
}

#### 2.4 Create User Vectors,
User_Vectoriser <- function(input_data) {
  
  #Parameters:
  #input_data : movielens dataset
  
  if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  
  # create output structure
  output <- vector(mode = "list", length = 0)

  # Split into individual genres and rank 
  genres <- input_data %>%
    select(genres) %>%
    group_by(genres) %>%
    summarise(n = n())
  
  genres <- genres %>%
    separate_rows(genres, sep = "\\|") %>%
    group_by(genres) %>%
    summarise(n_t = sum(n)) %>%
    arrange(-n_t)

  # calculate averages
  genres_list <- as.vector(genres$genres)
  #first calculation
  user_averages <- input_data %>%
    select(userId, rating, genres) %>%
    filter(grepl(genres_list[1], genres)) %>%
    group_by(userId) %>%
    summarise(avg_rating = mean(rating)) %>%
    mutate(genre = genres_list[1])

  #repeat for reamining genres
  v <- 2:nrow(genres)
  for (i in v) {
    user_averages_i <- input_data %>%
      select(userId, rating, genres) %>%
      filter(grepl(genres_list[i], genres)) %>%
      group_by(userId) %>%
      summarise(avg_rating = mean(rating)) %>%
      mutate(genre = genres_list[i])

    user_averages <- rbind(user_averages, user_averages_i)
  }

  
# formatting
  user_averages <- user_averages %>%
    spread(genre, avg_rating) %>%
    select(userId, genres_list)

# calculate weights per genre
  
  user_n <- input_data %>%
    select(userId, movieId, genres) %>%
    filter(grepl(genres_list[1], genres)) %>%
    group_by(userId) %>%
    summarise(n = n()) %>%
    mutate(genre = genres_list[1])

  for (i in v) {
    user_n_i <- input_data %>%
      select(userId, genres) %>%
      filter(grepl(genres_list[i], genres)) %>%
      group_by(userId) %>%
      summarise(n = n()) %>%
      mutate(genre = genres_list[i])

    user_n <- rbind(user_n, user_n_i)
  }

  user_n_i <- user_n %>%
    group_by(userId) %>%
    summarise(n_t = sum(n))
  
  user_n <- user_n %>%
    left_join(user_n_i, by = "userId") %>%
    mutate(n = n / n_t) %>%
    select(-n_t)
  
  user_n <- user_n %>%
    spread(genre, n) %>%
    select(userId, genres_list)

  # replace NAs with zeros
  
  user_averages[2:ncol(user_averages)][is.na(user_averages[2:ncol(user_averages)])] <- 0
  user_n[2:ncol(user_n)][is.na(user_n[2:ncol(user_n)])] <- 0


  # generate output vector
  output$genres <- genres
  output$user_averages <- user_averages
  output$user_weights <- user_n

  output
}

#### 2.5 Classify (Cluster) users,
User_Classifier <- function(input_data, 
                            user_vector = 1,
                            genres = 4,
                            step_cutoff = 0.5,
                            cluster_n = 3, 
                            cluster_type = "GMM", 
                            iterations = 2) {

  #Parameters:
  # input_data : movielens "tidied-up" dataset
  # user_vector : output of User_Vectoriser function
  # genres : number of genres used for clustering (sorted by popularity)
  # step_cutoff: iteration threshold, compared against sum percentages for selected genres.
  # cluster_n: Target number of clusters per run, parameter for clustering algorithms
  # cluster_type: Whether to use "GMM" or "kmeans" method
  # iterations: Number of times to iterate this process.
  # Please note that maximum total number of clusters will be cluster_n*iterations

  
  # Calculate user vectors if not provided

  if (!(typeof(user_vector) == "list")) {
    user_vector <- User_Vectoriser(input_data)
  }

  #Run if only one iteration
  
  if (iterations < 2) {
    
    #Create vector for clustering, composed of averages and genre weights, according number of desired genres.
    
    user_profiles <- user_vector$user_averages[, 0:genres + 1] %>%
      left_join(user_vector$user_weights[, 0:genres + 1], by = "userId")
    us <- user_profiles %>% select(-userId)
    
    # Cluster, using either kmeans or GMM
    
    if (cluster_type == "kmeans") {
      users_classification <- kmeans(us, centers = cluster_n, nstart = 30, iter.max = 30)
      user_profiles$group <- users_classification$cluster
      rm(users_classification, us)
    } else {
      fit <- GMM(us, cluster_n, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10, em_iter = 10, verbose = F)
      pr <- predict_GMM(us, fit$centroids, fit$covariance_matrices, fit$weights)
      user_profiles$group <- pr$cluster_labels
      rm(fit, pr, us)
    }
    #Result : user/cluster mapping
    
    user_profiles <- user_profiles %>% select(userId, group)
  } else {
    
    #two or more iterations 
    
    #Create vector for clustering, composed of averages and genre weights, according number of desired genres.
    
    user_profiles <- user_vector$user_averages[, 0:genres + 1] %>%
      left_join(user_vector$user_weights[, 0:genres + 1], by = "userId")

    us <- user_profiles %>% select(-userId)

    # first iteration

    if (cluster_type == "kmeans") {
      users_classification <- kmeans(us, centers = cluster_n, nstart = 30, iter.max = 30)
      user_profiles$group <- users_classification$cluster
      rm(users_classification, us)
    } else {
      fit <- GMM(us, cluster_n, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10, em_iter = 10, verbose = F)
      pr <- predict_GMM(us, fit$centroids, fit$covariance_matrices, fit$weights)
      user_profiles$group <- pr$cluster_labels
      rm(fit, pr, us)
    }

    # prepare  for second iteration

    user_vector_i <- user_vector
    user_vector_i$user_weights$percentage <- rowSums(user_vector_i$user_weights[, 2:genres + 1])

    excluded_ids <- user_vector_i$user_weights %>%
      filter(percentage >= step_cutoff) %>%
      select(userId)
    excluded_lines <- input_data %>% filter(userId %in% excluded_ids$userId)

    # cut excluded from user_profiles
    user_profiles <- user_profiles %>%
      filter(!(userId %in% excluded_ids$userId)) %>%
      select(userId, group)

    #iterate 
    
    it <- 2:iterations

    for (i in it) {
      if (nrow(excluded_lines) > 0) {
        user_vector_i <- User_Vectoriser(excluded_lines)

        genres_i <- nrow(user_vector_i$genres) - genres - 1
        if (genres_i > 0) {
          if (genres_i > genres) {
            genres_i <- genres
          }

          user_profiles_i <- user_vector_i$user_averages[, 0:genres_i + 1] %>%
            left_join(user_vector_i$user_weights[, 0:genres_i + 1], by = "userId")

          us_i <- user_profiles_i %>% select(-userId)

          if (cluster_type == "kmeans") {
            users_classification <- kmeans(us_i, centers = cluster_n, nstart = 30, iter.max = 30)
            user_profiles_i$group <- users_classification$cluster + (i) * 100
            rm(users_classification, us_i)
          }
          else {
            fit <- GMM(us_i, cluster_n, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10, em_iter = 10, verbose = F)
            if (typeof(fit$Error) == "list") {
              user_profiles_i$group <- -1
            } else {
              pr <- predict_GMM(us_i, fit$centroids, fit$covariance_matrices, fit$weights)
              user_profiles_i$group <- pr$cluster_labels + (i) * 100
            }
            rm(fit, pr, us_i)
          }

          user_profiles_i <- user_profiles_i %>% select(userId, group)
          user_profiles <- rbind(user_profiles, user_profiles_i)
          rm(user_profiles_i)
        }
      }

      # prep next iteration

      user_vector_i$user_weights$percentage <- rowSums(user_vector_i$user_weights[, 2:genres + 1])

      excluded_ids <- user_vector_i$user_weights %>%
        filter(percentage <= step_cutoff) %>%
        select(userId)
      excluded_lines <- input_data %>% filter(userId %in% excluded_ids$userId)
    }
  }
  
  # catch any excluded items
  excluded_lines <- input_data %>%
    filter(!(userId %in% user_profiles$userId)) %>%
    select(userId) %>%
    mutate(group = -1)
  user_profiles <- rbind(user_profiles, excluded_lines)

  user_profiles
}

#### 2.6 Calculate biases for general model.
Group_Biases <- function(input_data, user_profiles,
                         lambda_1 = 0, lambda_2 = 0) {
  
  
  output <- vector(mode = "list", length = 0)

  training_modified <- input_data %>% left_join(user_profiles, by = "userId")

  ## calculate averages per cluster group
  output$group_avgs <- training_modified %>%
    select(group, rating) %>%
    group_by(group) %>%
    summarise(n_group = n(), mu_group = mean(rating), sd_group = sd(rating))

  ## movie biases per cluster group
  output$movie_group_avgs <- training_modified %>%
    left_join(output$group_avgs, by = c("group")) %>%
    group_by(group, movieId) %>%
    summarise(b_m = sum(rating - mu_group) / (n() + lambda_1), s_m = sd(rating - mu_group), mu_group = max(mu_group))

  ## user biases per cluster group

  output$user_group_avgs <- training_modified %>%
    left_join(output$movie_group_avgs, by = c("movieId", "group")) %>%
    group_by(userId) %>%
    summarise(b_ug = sum(rating - mu_group - b_m) / (n() + lambda_2), s_ug = sd(rating - mu_group - b_m))

  ### cleanup
  rm(training_modified)

  output
}

#### 2.7 Calculate biases for cluster-based model
General_Biases <- function(input_data, lambda_3 = 0, lambda_4 = 0) {
  output <- vector(mode = "list", length = 0)

  mu_hat <- mean(input_data$rating)

  output$mu_hat <- mu_hat
  ### calculate movie and user bias

  movie_avgs <- input_data %>%
    group_by(movieId) %>%
    summarize(n_i = n(), b_i = sum(rating - mu_hat) / (n() + lambda_3), s_i = sd(rating - mu_hat))

  user_avgs <- input_data %>%
    left_join(movie_avgs, by = "movieId") %>%
    group_by(userId) %>%
    summarize(n_u = n(), b_u = sum(rating - b_i - mu_hat) / (n() + lambda_4), s_u = sd(rating - mu_hat - b_i))

  ### select and cleanup remnant test set
  output$movie_avgs <- movie_avgs
  output$user_avgs <- user_avgs
  output
}

#### 2.8 Predict Rating 
Rating_Predicter <- function(input_data, user_clustering, 
                             cluster_biases,
                             general_biases, 
                             s_m_threshold = FALSE, 
                             s_i_threshold = FALSE, 
                             calculate_rmse = TRUE) {

  ### declare output list
  output <- vector(mode = "list", length = 0)

  data_set <- input_data

  ### Method 1 - Use cluster  Parameters ###

  ### Remove "-1" group, which contains  users that couldnt be clustered

  data_set <- data_set %>%
    left_join(user_clustering, by = "userId") %>%
    filter(!(group == -1)) %>%
    filter(!is.na(group))

  #### add movie and user biases

  data_set <- data_set %>%
    left_join(cluster_biases$movie_group_avgs, by = c("movieId", "group")) %>%
    left_join(cluster_biases$user_group_avgs, by = "userId")

  ### filter only if threshold is a number (not default)

  if (is.numeric(s_m_threshold) == TRUE) {
    data_set <- data_set %>% filter(s_m <= s_m_threshold)
  }

  ### calculate prediction, filter results where prediction is NA

  prediction_1 <- data_set %>%
    mutate(predicted_rating = mu_group + b_m + b_ug, error = predicted_rating - rating, method = 1) %>%
    filter(!is.na(predicted_rating))

  #### Method 2 - Use General Parameters

  ### filter data set not covered in previous iteration

  data_set <- input_data %>% filter(!(row_id %in% prediction_1$row_id))

  ### add biases and filter by s_i threshold

  data_set <- data_set %>%
    left_join(general_biases$movie_avgs, by = "movieId") %>%
    left_join(general_biases$user_avgs, by = "userId")


  ### filter only if threshold is a number (not default)

  if (is.numeric(s_i_threshold) == TRUE) {
    data_set <- data_set %>% filter(s_m <= s_i_threshold)
  }


  ### create predictions

  prediction_2 <- data_set %>%
    mutate(predicted_rating = general_biases$mu_hat + b_i + b_u, error = rating - predicted_rating, method = 2) %>%
    filter(!(predicted_rating == general_biases$mu_hat))

  #### Method 3 - just guess

  #### filter from previous data set

  data_set <- input_data %>%
    filter(!(row_id %in% prediction_1$row_id)) %>%
    filter(!(row_id %in% prediction_2$row_id))

  ### create predictions assuming that 80% of all results are within two standard deviations of the average (for a normal distribution)

  sd_hat <- sd(input_data$rating)

  data_set$random_number <- runif(nrow(data_set), -2, 2)

  prediction_3 <- data_set %>%
    mutate(predicted_rating = general_biases$mu_hat + (sd_hat) * random_number) %>%
    mutate(error = rating - predicted_rating, method = 3)

  #### put everything together and create output

  prediction_1 <- prediction_1 %>% select(userId, movieId, title, release_year, timestamp, rating, method, predicted_rating)
  prediction_2 <- prediction_2 %>% select(userId, movieId, title, release_year, timestamp, rating, method, predicted_rating)
  prediction_3 <- prediction_3 %>% select(userId, movieId, title, release_year, timestamp, rating, method, predicted_rating)

  prediction <- rbind(prediction_1, prediction_2)
  prediction <- rbind(prediction, prediction_3)

  output$prediction <- prediction

  output$distribution_numbers$"1" <- nrow(prediction_1)
  output$distribution_numbers$"2" <- nrow(prediction_2)
  output$distribution_numbers$"3" <- nrow(prediction_3)

  output$distribution_percentage$"1" <- output$distribution_numbers$"1" / nrow(prediction)
  output$distribution_percentage$"2" <- output$distribution_numbers$"2" / nrow(prediction)
  output$distribution_percentage$"3" <- output$distribution_numbers$"3" / nrow(prediction)

  if (calculate_rmse == TRUE) {
    output$RMSE$overall <- RMSE(prediction$rating, prediction$predicted_rating)
    output$RMSE$"1" <- RMSE(prediction_1$rating, prediction_1$predicted_rating)
    output$RMSE$"2" <- RMSE(prediction_2$rating, prediction_2$predicted_rating)
    output$RMSE$"3" <- RMSE(prediction_3$rating, prediction_3$predicted_rating)
  }

  output
}

#### 2.9 Load data from Movielens source (code provided in course materials)
Movielens_Data_Loader <- function() {
  output <- vector(mode = "list", length = 0)

  ################################
  # Create edx set, validation set
  ################################

  # Note: this process could take a couple of minutes

  if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if (!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip

  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

  ratings <- fread(
    text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
    col.names = c("userId", "movieId", "rating", "timestamp")
  )

  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(
    movieId = as.numeric(levels(movieId))[movieId],
    title = as.character(title),
    genres = as.character(genres)
  )

  movielens <- left_join(ratings, movies, by = "movieId")

  # Validation set will be 10% of MovieLens data
  set.seed(1, sample.kind = "Rounding")
  # if using R 3.5 or earlier, use `set.seed(1)` instead
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index, ]
  temp <- movielens[test_index, ]

  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>%
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")

  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)

  rm(dl, ratings, movies, test_index, temp, movielens, removed)

#### SMALL addition to code provided , so it can run as a function returning the edx and validation datasets.  
  output$edx <- edx
  output$validation <- validation
}

#####################################################################################################
#####################################################################################################
# 3. Code to run prediction with optimal parameters (commented)
### This code is mean tot be ran AFTER TRAINING - therefore there is no further split for training 
#and testing data.

#Optimal parameters
#cluster_n <-20
#cluster_type<-"GMM"
#clustering_iterations <-2
#genres <- 6
#step_cutoff <- 0.5
#lambda_1 <- 2
#lambda_2 <- 4.5

### Download data from the Internet

## Uncomment the below lines to reload the Movielens database
#movielens_10M<-Movielens_Data_Loader()
#edx <- movielens_10M$edx
#validation <- movielens_10M$validation
#rm(movielens_10M)

##Tidy up edx data for training.
#edx<- Tidy_Up(edx)

### Retrain : reclustering and re-calculation of biases with new dataset

#user_vector_prediction <-User_Vectoriser(edx)
#general_biases_prediction <- General_Biases(edx)
#user_classification_prediction  <- User_Classifier(edx,user_vector_prediction,
#                                                   genres=genres,
#                                                   step_cutoff=step_cutoff,
#                                                   cluster_n=cluster_n,
#                                                   cluster_type=cluster_type,
#                                                   iterations=clustering_iterations)
#biases_by_group_prediction <- Group_Biases(edx,
#                                           user_classification_prediction,
#                                           lambda_1=lambda_1,
#                                           lambda_2=lambda_2)

### tidy up validation table

#validation<-Tidy_Up(validation)

# Predict ratings

#prediction <- Rating_Predicter(validation,
#                               user_classification_prediction,
#                               biases_by_group_prediction,
#                               general_biases_prediction)


##Calculate method split and RMSE
#prediction_stats <- tibble(RMSE = double(),RMSE_1 = double(),
#                RMSE_2 = double(),
#                method_1=double(),method_2=double())

#prediction_stats <- tibble(RMSE = prediction$RMSE$overall,
#                RMSE_1=prediction$RMSE$`1`,
#                RMSE_2=prediction$RMSE$`2`,
#                method_1=prediction$distribution_percentage$`1`,
#                method_2=prediction$distribution_percentage$`2`)

#####################################################################################################
#####################################################################################################
#End of File
#####################################################################################################
#####################################################################################################
