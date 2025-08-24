GLV <- function(t, x, parameters){
  with(as.list(c(x, parameters)), {
    x[x < 10^-5] <- 0 
    dxdt <- x * (r + A %*% x)
    list(dxdt)
  })
}


integrate_GLV <- function(r, A, x0, t0, maxtime = 50, steptime = 0.1){
  times <- seq(t0, maxtime, by = steptime)
  parameters <- list(r = r, A = A)
  out <- ode(y = x0, times = times, 
             func = GLV, parms = parameters, 
             method = "ode45")

  return(out)
}

