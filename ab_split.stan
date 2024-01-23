data {
  int<lower=0> N1;
  int<lower=0> N2;
  vector[N1] x1;
  vector[N2] x2;
}

parameters {
  real x1mu;
  real x2mu;
  real<lower=0> x1sigma;
  real<lower=0> x2sigma;
}

model {
  x1 ~ normal(x1mu, x1sigma);
  x2 ~ normal(x2mu, x2sigma);
}

generated quantities {
  real meandiff;
  meandiff = x1mu - x2mu;
}
