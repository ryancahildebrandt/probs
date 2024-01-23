data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> ysigma;
}

model {
  y ~ normal(beta0 + x * beta1, ysigma);
}

