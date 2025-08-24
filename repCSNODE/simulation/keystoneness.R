rm(list = ls())
library(reticulate)
source("function/generate_replicator.R")
library(deSolve)

pickle <- import("pickle")
py <- import_builtins()



extract_trajectory_single <- function(time_traj, obs_time) {
  obs_time_rounded <- round(obs_time, 3)

  time_rounded <- round(time_traj[, 1], 3)

  traj_discret <- time_traj[time_rounded %in% obs_time_rounded, -1]

  return(traj_discret)
}


process_zeroed_trajectories <- function(A, x0, obs_time, integ_steptime) {

  obs_time <- obs_time - min(obs_time)

  t0 <- min(obs_time)

  modified_time_traj <- integrate_replicator(A, x0, t0 = t0, maxtime = max(obs_time), steptime = integ_steptime)

  zeroed_obs_traj <- extract_trajectory_single(modified_time_traj, obs_time)

  return(zeroed_obs_traj)
}



node_process_main <- function(d) {
  data_folder <- "data_generate"
  file_path <- file.path(data_folder, "results_repcsnode.pkl")

  results <- py_load_object(file_path)

  select_subject_obs <- results$select_subject_obs
  obs_time <- results$obs_time



  A_node <- results$A_node

  integ_steptime <- 0.01

  result_df <- data.frame(
    Keystoneness = numeric(),
    stringsAsFactors = FALSE
  )

  colnames(select_subject_obs) <- paste0("species", 1:ncol(select_subject_obs))


  test_data <- select_subject_obs
  test_seq_time <- obs_time


  for (col_name in colnames(test_data)) {
    modified_data <- test_data
    modified_data <- as.data.frame(modified_data)


    p_col <- modified_data[[col_name]]

    modified_data[[col_name]] <- 0
    modified_data <- as.data.frame(modified_data)
    test_data_normalized <- sweep(modified_data, 1, rowSums(modified_data), "/")

    x0 <- as.numeric(test_data_normalized[1, ])


    zeroed_col <- process_zeroed_trajectories(
      A = A_node,
      x0 = x0,
      obs_time = test_seq_time,
      integ_steptime = integ_steptime
    )

    colnames(zeroed_col) <- colnames(test_data_normalized)

    other_columns <- setdiff(colnames(test_data_normalized), col_name)
    test_data_normalized_without_col <- test_data_normalized[, other_columns, drop = FALSE]
    zeroed_col_without_col <- zeroed_col[, other_columns, drop = FALSE]

    test_data_normalized_without_col_matrix <- as.matrix(test_data_normalized_without_col)
    zeroed_col_matrix <- as.matrix(zeroed_col_without_col)

    se_values <- numeric(nrow(test_data_normalized))

    for (i in 1:nrow(zeroed_col_matrix)) {
      row_diff <- test_data_normalized_without_col_matrix[i, ] - zeroed_col_matrix[i, ]
      se <- sum((row_diff)^2) * (1 - p_col[i]) / sum(1 - p_col)
      se_values[i] <- se
    }

    final_result <- sqrt(sum(se_values) / (dim(zeroed_col_matrix)[1] * dim(zeroed_col_matrix)[2]))


    result_df <- rbind(result_df, data.frame(
      taxonomy = col_name,
      Keystoneness = final_result
    ))
  }



  result_df <- result_df[order(result_df$Keystoneness, decreasing = TRUE), ]


  file_name <- paste0(data_folder, "/keystoneness.csv")


  write.csv(result_df, file = file_name, row.names = FALSE)
}


d <- 15



node_process_main(d)

