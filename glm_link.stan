data {
  int<lower=0> N;
  vector[N] x;
  int<lower=0,upper=1> y[N];
}

parameters {
  real beta0;
  real beta1;
}

model {
  y ~ bernoulli_logit(beta0 + beta1 * x);
}

