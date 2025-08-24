replicator <- function(t, x, parameters) {
  with(as.list(c(x, parameters)), {


    fitness <- A %*% x
    
    avg_fitness <- sum(x * fitness)
    
    dxdt <- x * (fitness - avg_fitness)
    
    list(dxdt)
  })
}


integrate_replicator <- function(A, x0, t0, maxtime = 50, steptime = 0.1){
  times <- seq(t0, maxtime, by = steptime)
  parameters <- list(A = A)
  
  event_func <- function(t, state, parms) {
    state <- state / sum(state)  
    return(state)
  }

  out <- ode(y = x0, times = times, 
             func = replicator, parms = parameters, 
             method = "ode45", events = list(func = event_func, time = times))
  
  return(out)
}

