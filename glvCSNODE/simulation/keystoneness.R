rm(list=ls())
library(reticulate)
source("function/generate_glv.R")
library(deSolve)



pickle <- import("pickle")
py <- import_builtins()



extract_trajectory_single <- function(time_traj, obs_time) {
  obs_time_rounded <- round(obs_time, 3)
  
  time_rounded <- round(time_traj[, 1], 3)
  
  traj_discret <- time_traj[time_rounded %in% obs_time_rounded, -1]

  if (nrow(traj_discret) != length(obs_time_rounded)) {
    stop(paste0(
      "Mismatch detected: traj_discret rows (", nrow(traj_discret), 
      ") do not match obs_time_rounded length (", length(obs_time_rounded), ")."
    ))
  }
  
  return(traj_discret)
}


process_zeroed_trajectories <- function(A, r, x0, obs_time, integ_steptime) {
  
  obs_time <- obs_time - min(obs_time)
  t0 <- min(obs_time)

  result <- tryCatch({
    modified_time_traj <- integrate_GLV(r, A, x0, t0, maxtime = max(obs_time), steptime = integ_steptime)
    zeroed_obs_traj <- extract_trajectory_single(modified_time_traj, obs_time)
    zeroed_obs_traj
  }, error = function(e) {
    matrix(NaN, nrow = length(obs_time), ncol = length(x0))
  })

  return(result)
}



node_process_main <- function(d) {

  data_folder <- 'data_generate'
  file_path <- file.path(data_folder, 'results_glvcsnode.pkl')


  integ_steptime <- 0.01

  result_df <- data.frame(
    taxonomy = character(),
    Keystoneness = numeric()
  )



  results <- py_load_object(file_path)
  A_node <- results$A_node
  r_node <- results$r_node

  r_node <- matrix(r_node, ncol = 1)

  select_subject_obs <- results$select_subject_obs
  obs_time <- results$obs_time

  colnames(select_subject_obs) <- paste0("species", 1:ncol(select_subject_obs))


  test_data <- select_subject_obs

  test_data <- as.data.frame(test_data)

  test_time <- obs_time

  for (taxonomy in colnames(test_data)) {
    
    modified_data <- test_data

    modified_data[taxonomy] <- 0

    modified_data_normalized <- sweep(modified_data, 1, rowSums(modified_data), "/")

    normalized_test_data <- sweep(test_data, 1, rowSums(test_data), "/")

    p_taxonomy <- normalized_test_data[[taxonomy]]

    x0 <- as.numeric(modified_data[1, ])

    zeroed_taxonomy <- process_zeroed_trajectories(
      A = A_node,
      r = r_node,
      x0 = x0,
      obs_time = test_time,
      integ_steptime = integ_steptime
    )

    zeroed_taxonomy <- sweep(zeroed_taxonomy, 1, rowSums(zeroed_taxonomy), "/")

    colnames(zeroed_taxonomy) <- colnames(modified_data)

    other_taxonomies <- setdiff(colnames(modified_data), taxonomy)

    modified_data_normalized_without_taxonomy <- modified_data_normalized[, other_taxonomies, drop = FALSE]
    zeroed_taxonomy_without_taxonomy <- zeroed_taxonomy[, other_taxonomies, drop = FALSE]

    modified_matrix <- as.matrix(modified_data_normalized_without_taxonomy)
    zeroed_taxonomy_matrix <- as.matrix(zeroed_taxonomy_without_taxonomy)

    se_values <- numeric(nrow(modified_matrix))

    for (i in 1:nrow(modified_matrix)) {
      row_diff <- modified_matrix[i, ] - zeroed_taxonomy_matrix[i, ]
      se <- sum((row_diff)^2) * (1 - p_taxonomy[i]) / sum(1 - p_taxonomy)
      se_values[i] <- se
    }

    final_result <- sqrt(sum(se_values) / (dim(modified_data_normalized_without_taxonomy)[1] * dim(modified_data_normalized_without_taxonomy)[2]))

    result_df <- rbind(result_df, data.frame(
      taxonomy = taxonomy,
      Keystoneness = final_result
    ))
  }
      
    
      

  result_df <- result_df[order(result_df$Keystoneness, decreasing = TRUE), ]


  file_name <- file.path(data_folder, "keystoneness.csv")

  write.csv(result_df, file = file_name, row.names = FALSE)  

  }


d <- 15


node_process_main(d)