data {
  int<lower=1> K;
  int<lower=1> N;
  vector[N] x;
}

parameters {
  ordered[K] mu;
  vector<lower=0>[K] sigma;
  simplex[K] theta;
}

model {
  mu ~ normal(0, 100);
  sigma ~ normal(0, 20);
  vector[K] log_theta = log(theta);

  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K) {
      lps[k] += normal_lpdf(x[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}
