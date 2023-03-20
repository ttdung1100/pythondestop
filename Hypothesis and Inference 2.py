def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
# if x is greater than the mean, the tail is what's greater than x
    return 2 * normal_probability_above(x, mu, sigma)
        else:
# if x is less than the mean, the tail is what's less than x
    return 2 * normal_probability_below(x, mu, sigma)
    
two_sided_p_value(529.5, mu_0, sigma_0) # 0.062

extreme_value_count = 0
for _ in range(100000):
num_heads = sum(1 if random.random() < 0.5 else 0 # count # of heads
for _ in range(1000)) # in 1000 flips
if num_heads >= 530 or num_heads <= 470: # and count how often

    extreme_value_count += 1 # the # is 'extreme'
print extreme_value_count / 100000 # 0.062

two_sided_p_value(531.5, mu_0, sigma_0) # 0.0463    

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_p_value(524.5, mu_0, sigma_0) # 0.061

upper_p_value(526.5, mu_0, sigma_0) # 0.047

math.sqrt(p * (1 - p) / 1000)

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158

normal_two_sided_bounds(0.95, mu, sigma) # [0.4940, 0.5560]

p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709]

#P-hacking
def run_experiment():
"""flip a fair coin 1000 times, True = heads, False = tails"""
    return [random.random() < 0.5 for _ in range(1000)]
def reject_fairness(experiment):
"""using the 5% significance levels"""
num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531
    random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
for experiment in experiments
if reject_fairness(experiment)])
print num_rejections # 46

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
        return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
        return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

z = a_b_test_statistic(1000, 200, 1000, 180) # -1.14

two_sided_p_value(z) # 0.254

z = a_b_test_statistic(1000, 200, 1000, 150) # -2.94
two_sided_p_value(z) # 0.003

#Bayesian Inference
def B(alpha, beta):
"""a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1: # no weight outside of [0, 1]
        return 0
        return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

alpha / (alpha + beta)



mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)


mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

normal_two_sided_bounds(0.95, mu_0, sigma_0) # (469, 531)



# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.887

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability # 0.936




upper_bound = normal_lower_bound(tail_probability, mu, sigma)
lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return upper_bound , lower_bound





