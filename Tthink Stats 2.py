#Tthink Stats 2
a = range(35.46)

weeks = range(35, 46)
diffs = []
for week in weeks:
    p1 = first_pmf.Prob(week)
    p2 = other_pmf.Prob(week)
    diff = 100 * (p1 - p2)
    diffs.append(diff)
    
thinkplot.Bar(weeks, diffs)
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(first_pmf, align='right', width=width)
thinkplot.Hist(other_pmf, align='left', width=width)
thinkplot.Config(xlabel='weeks',ylabel='probability',axis=[27, 46, 0, 0.6])
thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([first_pmf, other_pmf])
thinkplot.Show(xlabel='weeks',axis=[27, 46, 0, 0.6])







d = { 7: 8, 12: 8, 17: 14, 22: 4, 27: 6, 32: 12, 37: 8, 42: 3, 47: 2 }
pmf = thinkstats2.Pmf(d, label='actual')
print('mean', pmf.Mean())



def BiasPmf(pmf, label):
        new_pmf = pmf.Copy(label=label)
    for x, p in pmf.Items():
            new_pmf.Mult(x, x)
            new_pmf.Normalize()
            return new_pmf


biased_pmf = BiasPmf(pmf, label='observed')

thinkplot.PrePlot(2)




thinkplot.Pmfs([pmf, biased_pmf])
thinkplot.Show(xlabel='class size', ylabel='PMF')



def UnbiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)
    for x, p in pmf.Items():
        new_pmf.Mult(x, 1.0/x)
        new_pmf.Normalize()
        return new_pmf



def PercentileRank(scores, your_score):
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1
            percentile_rank = 100.0 * count / len(scores)
            return percentile_rank



def Percentile(scores, percentile_rank):
    scores.sort()
    for score in scores:
        if PercentileRank(scores, score) >= percentile_rank:
            return score



def Percentile2(scores, percentile_rank):
    scores.sort()
    index = percentile_rank * (len(scores)-1) // 100
    return scores[index]


def EvalCdf(sample, x):
    count = 0.0
    for value in sample:
        if value <= x:
            count += 1
            prob = count / len(sample)
            return prob


live, firsts, others = first.MakeFrames()
cdf = thinkstats2.Cdf(live.prglngth, label='prglngth')


thinkplot.Cdf(cdf)
thinkplot.Show(xlabel='weeks', ylabel='CDF')



first_cdf = thinkstats2.Cdf(firsts.totalwgt_lb, label='first')
other_cdf = thinkstats2.Cdf(others.totalwgt_lb, label='other')
thinkplot.PrePlot(2)
thinkplot.Cdfs([first_cdf, other_cdf])
thinkplot.Show(xlabel='weight (pounds)', ylabel='CDF')


weights = live.totalwgt_lb
cdf = thinkstats2.Cdf(weights, label='totalwgt_lb')


sample = np.random.choice(weights, 100, replace=True)
ranks = [cdf.PercentileRank(x) for x in sample]



rank_cdf = thinkstats2.Cdf(ranks)
thinkplot.Cdf(rank_cdf)
thinkplot.Show(xlabel='percentile rank', ylabel='CDF')



# class Cdf:
def Random(self):
    return self.Percentile(random.uniform(0, 100))




def PositionToPercentile(position, field_size):
    beat = field_size - position + 1
    percentile = 100.0 * beat / field_size
    return percentile



def PercentileToPosition(percentile, field_size):
    beat = percentile * field_size / 100.0
    position = field_size - beat + 1
    return position


df = ReadBabyBoom()
diffs = df.minutes.diff()
cdf = thinkstats2.Cdf(diffs, label='actual')
thinkplot.Cdf(cdf)
thinkplot.Show(xlabel='minutes', ylabel='CDF')



thinkplot.Cdf(cdf, complement=True)
thinkplot.Show(xlabel='minutes',
ylabel='CCDF',
yscale='log')


def EvalNormalCdf(x, mu=0, sigma=1):
    return scipy.stats.norm.cdf(x, loc=mu, scale=sigma)



xs, ys = thinkstats2.NormalProbability(sample)


def MakeNormalPlot(weights):
mean = weights.mean()
std = weights.std()
xs = [-4, 4]
fxs, fys = thinkstats2.FitLine(xs, inter=mean, slope=std)
thinkplot.Plot(fxs, fys, color='gray', label='model')
xs, ys = thinkstats2.NormalProbability(weights)
thinkplot.Plot(xs, ys, label='birth weights')


def expovariate(lam):
    p = random.random()
    x = -math.log(1-p) / lam
    return x



class NormalPdf(Pdf):
def __init__(self, mu=0, sigma=1, label=''):
self.mu = mu
self.sigma = sigma
self.label = label
def Density(self, xs):
    return scipy.stats.norm.pdf(xs, self.mu, self.sigma)
    
def GetLinspace(self):
low, high = self.mu-3*self.sigma, self.mu+3*self.sigma
    return np.linspace(low, high, 101)



class EstimatedPdf(Pdf):
def __init__(self, sample):
self.kde = scipy.stats.gaussian_kde(sample)
def Density(self, xs):
    return self.kde.evaluate(xs)



def Incr(self, x, term=1):
    self.d[x] = self.d.get(x, 0) + term
    def Mult(self, x, factor):
        self.d[x] = self.d.get(x, 0) * factor
        def Remove(self, x):
            del self.d[x]



# class Pmf
def Normalize(self, fraction=1.0):
    total = self.Total()
    if total == 0.0:
        raise ValueError('Total probability is zero.')
    factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor
    return total


self.xs, freqs = zip(*sorted(dw.Items()))
self.ps = np.cumsum(freqs, dtype=np.float)
self.ps /= self.ps[-1]



# class Cdf
def Prob(self, x):
    if x < self.xs[0]:
        return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
    return p



# class Cdf
def Value(self, p):
    if p < 0 or p > 1:
    raise ValueError('p must be in range [0, 1]')
index = bisect.bisect_left(self.ps, p)
    return self.xs[index]



# class Cdf
def Items(self):
    a = self.ps
    b = np.roll(a, 1)
    b[0] = 0
    return zip(self.xs, a-b)

def CentralMoment(xs, k):
    mean = RawMoment(xs, 1)
    return sum((x - mean)**k for x in xs) / len(xs)



def StandardizedMoment(xs, k):
var = CentralMoment(xs, 2)
std = math.sqrt(var)
    return CentralMoment(xs, k) / std**k

def Skewness(xs):
    return StandardizedMoment(xs, 3)


def Median(xs):
cdf = thinkstats2.Cdf(xs)
    return cdf.Value(0.5)
    
def PearsonMedianSkewness(xs):
median = Median(xs)
mean = RawMoment(xs, 1)
var = CentralMoment(xs, 2)
std = math.sqrt(var)
gp = 3 * (mean - median) / std
    return gp


live, firsts, others = first.MakeFrames()
data = live.totalwgt_lb.dropna()
pdf = thinkstats2.EstimatedPdf(data)
thinkplot.Pdf(pdf, label='birth weight')



df = brfss.ReadBrfss(nrows=None)
data = df.wtkg2.dropna()
pdf = thinkstats2.EstimatedPdf(data)
thinkplot.Pdf(pdf, label='adult weight')    


df = brfss.ReadBrfss(nrows=None)
sample = thinkstats2.SampleRows(df, 5000)
heights, weights = sample.htm3, sample.wtkg2




def SampleRows(df, nrows, replace=False):
indices = np.random.choice(df.index, nrows, replace=replace)
sample = df.loc[indices]
    return sample




thinkplot.Scatter(heights, weights)
thinkplot.Show(xlabel='Height (cm)',
ylabel='Weight (kg)',
axis=[140, 210, 20, 200])




heights = thinkstats2.Jitter(heights, 1.3)



weights = thinkstats2.Jitter(weights, 0.5)



def Jitter(values, jitter=0.5):
n = len(values)
    return np.random.uniform(-jitter, +jitter, n) + values


thinkplot.Scatter(heights, weights, alpha=0.2)


thinkplot.HexBin(heights, weights)


df = df.dropna(subset=['htm3', 'wtkg2'])
bins = np.arange(135, 210, 5)
indices = np.digitize(df.htm3, bins)
groups = df.groupby(indices)

for i, group in groups:
print(i, len(group))




heights = [group.htm3.mean() for i, group in groups]
cdfs = [thinkstats2.Cdf(group.wtkg2) for i, group in groups]

for percent in [75, 50, 25]:
weights = [cdf.Percentile(percent) for cdf in cdfs]
label = '%dth' % percent
thinkplot.Plot(heights, weights, label=label)


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if meanx is None:
        meanx = np.mean(xs)
        if meany is None:
            meany = np.mean(ys)
            cov = np.dot(xs-meanx, ys-meany) / len(xs)
            return cov



def Corr(xs, ys):
xs = np.asarray(xs)
ys = np.asarray(ys)
meanx, varx = MeanVar(xs)
meany, vary = MeanVar(ys)
corr = Cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)
    return corr

def SpearmanCorr(xs, ys):
xranks = pandas.Series(xs).rank()
yranks = pandas.Series(ys).rank()
    return Corr(xranks, yranks)


def SpearmanCorr(xs, ys):
xs = pandas.Series(xs)
ys = pandas.Series(ys)
    return xs.corr(ys, method='spearman')

thinkstats2.Corr(df.htm3, np.log(df.wtkg2)))




def Estimate1(n=7, m=1000):
mu = 0
sigma = 1
means = []
medians = []
for _ in range(m):
xs = [random.gauss(mu, sigma) for i in range(n)]
xbar = np.mean(xs)
median = np.median(xs)
means.append(xbar)
medians.append(median)
print('rmse xbar', RMSE(means, mu))
print('rmse median', RMSE(medians, mu))




def RMSE(estimates, actual):
    e2 = [(estimate-actual)**2 for estimate in estimates]
mse = np.mean(e2)
    return math.sqrt(mse)

def Estimate2(n=7, m=1000):
mu = 0
sigma = 1
estimates1 = []
estimates2 = []
for _ in range(m):
xs = [random.gauss(mu, sigma) for i in range(n)]
biased = np.var(xs)
unbiased = np.var(xs, ddof=1)
estimates1.append(biased)
estimates2.append(unbiased)
print('mean error biased', MeanError(estimates1, sigma**2))
print('mean error unbiased', MeanError(estimates2, sigma**2))



def MeanError(estimates, actual):
errors = [estimate-actual for estimate in estimates]
    return np.mean(errors)


def SimulateSample(mu=90, sigma=7.5, n=9, m=1000):
means = []
for j in range(m):
xs = np.random.normal(mu, sigma, n)
xbar = np.mean(xs)
means.append(xbar)
cdf = thinkstats2.Cdf(means)
ci = cdf.Percentile(5), cdf.Percentile(95)
stderr = RMSE(means, mu)


def Estimate3(n=7, m=1000):
lam = 2
means = []
medians = []
for _ in range(m):
xs = np.random.exponential(1.0/lam, n)
L = 1 / np.mean(xs)
Lm = math.log(2) / thinkstats2.Median(xs)
means.append(L)
medians.append(Lm)
print('rmse L', RMSE(means, lam))
print('rmse Lm', RMSE(medians, lam))
print('mean error L', MeanError(means, lam))
print('mean error Lm', MeanError(medians, lam))



class HypothesisTest(object):
def __init__(self, data):
self.data = data

self.MakeModel()
self.actual = self.TestStatistic(data)
def PValue(self, iters=1000):
self.test_stats = [self.TestStatistic(self.RunModel())
for _ in range(iters)]
count = sum(1 for x in self.test_stats if x >= self.actual)
    return count / iters

def TestStatistic(self, data):
    raise UnimplementedMethodException()
def MakeModel(self):
    pass
def RunModel(self):
    raise UnimplementedMethodException()





class CoinTest(thinkstats2.HypothesisTest):
    def TestStatistic(self, data):
        heads, tails = data

test_stat = abs(heads - tails)
    return test_stat


def RunModel(self):
    heads, tails = self.data
    n = heads + tails
    sample = [random.choice('HT') for _ in range(n)]
    hist = thinkstats2.Hist(sample)
    data = hist['H'], hist['T']
    return data

ct = CoinTest((140, 110))
pvalue = ct.PValue()


class DiffMeansPermute(thinkstats2.HypothesisTest):
def TestStatistic(self, data):
group1, group2 = data
test_stat = abs(group1.mean() - group2.mean())
    return test_stat


def MakeModel(self):
group1, group2 = self.data
self.n, self.m = len(group1), len(group2)
self.pool = np.hstack((group1, group2))
def RunModel(self):
np.random.shuffle(self.pool)
data = self.pool[:self.n], self.pool[self.n:]
    return data


live, firsts, others = first.MakeFrames()
data = firsts.prglngth.values, others.prglngth.values
ht = DiffMeansPermute(data)
pvalue = ht.PValue()


ht.PlotCdf()
thinkplot.Show(xlabel='test statistic',
ylabel='CDF')


class DiffMeansOneSided(DiffMeansPermute):
def TestStatistic(self, data):
group1, group2 = data
test_stat = group1.mean() - group2.mean()
    return test_stat

class DiffStdPermute(DiffMeansPermute):
def TestStatistic(self, data):
group1, group2 = data
test_stat = group1.std() - group2.std()
    return test_stat


class CorrelationPermute(thinkstats2.HypothesisTest):
def TestStatistic(self, data):
xs, ys = data
test_stat = abs(thinkstats2.Corr(xs, ys))
    return test_stat

def RunModel(self):
xs, ys = self.data
xs = np.random.permutation(xs)
    return xs, ys




live, firsts, others = first.MakeFrames()
live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
data = live.agepreg.values, live.totalwgt_lb.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()



class DiceTest(thinkstats2.HypothesisTest):
def TestStatistic(self, data):
observed = data
n = sum(observed)
expected = np.ones(6) * n / 6
test_stat = sum(abs(observed - expected))
    return test_stat

def RunModel(self):
n = sum(self.data)
values = [1, 2, 3, 4, 5, 6]
rolls = np.random.choice(values, n, replace=True)
hist = thinkstats2.Hist(rolls)
freqs = hist.Freqs(values)
    return freqs



class DiceChiTest(DiceTest):
def TestStatistic(self, data):
observed = data
n = sum(observed)
expected = np.ones(6) * n / 6
test_stat = sum((observed - expected)**2 / expected)
    return test_stat




class PregLengthTest(thinkstats2.HypothesisTest):
    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))
        pmf = thinkstats2.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))
    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data


# class PregLengthTest:
def TestStatistic(self, data):
    firsts, others = data
    stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

def ChiSquared(self, lengths):
    hist = thinkstats2.Hist(lengths)
    observed = np.array(hist.Freqs(self.values))
    expected = self.expected_probs * len(lengths)
    stat = sum((observed - expected)**2 / expected)
    return stat


def FalseNegRate(data, num_runs=100):
    group1, group2 = data
    count = 0
    for i in range(num_runs):
        sample1 = thinkstats2.Resample(group1)
        sample2 = thinkstats2.Resample(group2)
        ht = DiffMeansPermute((sample1, sample2))
        pvalue = ht.PValue(iters=101)
        if pvalue > 0.05:
            count += 1
            return count / num_runs


def Resample(xs):
    return np.random.choice(xs, len(xs), replace=True)


live, firsts, others = first.MakeFrames()
data = firsts.prglngth.values, others.prglngth.values
neg_rate = FalseNegRate(data)

def LeastSquares(xs, ys):
meanx, varx = MeanVar(xs)
meany = Mean(ys)
slope = Cov(xs, ys, meanx, meany) / varx
inter = meany - slope * meanx
    return inter, slope


def FitLine(xs, inter, slope):
fit_xs = np.sort(xs)
fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys



live, firsts, others = first.MakeFrames()
live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
ages = live.agepreg
weights = live.totalwgt_lb
inter, slope = thinkstats2.LeastSquares(ages, weights)
fit_xs, fit_ys = thinkstats2.FitLine(ages, inter, slope)



def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res


def SamplingDistributions(live, iters=101):
    t = []
    for _ in range(iters):
        sample = thinkstats2.ResampleRows(live)
        ages = sample.agepreg
        weights = sample.totalwgt_lb
        estimates = thinkstats2.LeastSquares(ages, weights)
        t.append(estimates)
        inters, slopes = zip(*t)
            return inters, slopes



def ResampleRows(df):
    return SampleRows(df, len(df), replace=True)




def Summarize(estimates, actual=None):
mean = thinkstats2.Mean(estimates)
stderr = thinkstats2.Std(estimates, mu=actual)
cdf = thinkstats2.Cdf(estimates)
ci = cdf.ConfidenceInterval(90)
print('mean, SE, CI', mean, stderr, ci)



def PlotConfidenceIntervals(xs, inters, slopes,
    percent=90, **options):
    fys_seq = []
    for inter, slope in zip(inters, slopes):
        fxs, fys = thinkstats2.FitLine(xs, inter, slope)
        fys_seq.append(fys)
        p = (100 - percent) / 2
        percents = p, 100 - p
        
        low, high = thinkstats2.PercentileRows(fys_seq, percents)
        thinkplot.FillBetween(fxs, low, high, **options)



def CoefDetermination(ys, res):
    return 1 - Var(res) / Var(ys)


class SlopeTest(thinkstats2.HypothesisTest):
def TestStatistic(self, data):
ages, weights = data
_, slope = thinkstats2.LeastSquares(ages, weights)
    return slope
def MakeModel(self):
_, weights = self.data
self.ybar = weights.mean()
self.res = weights - self.ybar
def RunModel(self):
ages, _ = self.data
weights = self.ybar + np.random.permutation(self.res)
    return ages, weights



live, firsts, others = first.MakeFrames()
live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
ht = SlopeTest((live.agepreg, live.totalwgt_lb))
pvalue = ht.PValue()



inters, slopes = SamplingDistributions(live, iters=1001)
slope_cdf = thinkstats2.Cdf(slopes)
pvalue = slope_cdf[0]



def ResampleRowsWeighted(df, column='finalwgt'):
weights = df[column]
cdf = Cdf(dict(weights))
indices = cdf.Sample(len(weights))
sample = df.loc[indices]
    return sample


estimates = [ResampleRows(live).totalwgt_lb.mean()
for _ in range(iters)]



estimates = [ResampleRowsWeighted(live).totalwgt_lb.mean()
for _ in range(iters)]



import statsmodels.formula.api as smf
live, firsts, others = first.MakeFrames()
formula = 'totalwgt_lb ~ agepreg'
model = smf.ols(formula, data=live)
results = model.fit()





inter = results.params['Intercept']
slope = results.params['agepreg']




slope_pvalue = results.pvalues['agepreg']



print(results.summary())


diff_weight = firsts.totalwgt_lb.mean() - others.totalwgt_lb.mean()

diff_age = firsts.agepreg.mean() - others.agepreg.mean()

results = smf.ols('totalwgt_lb ~ agepreg', data=live).fit()

results = smf.ols('totalwgt_lb ~ agepreg', data=live).fit()

slope = results.params['agepreg']

slope = results.params['agepreg']
slope * diff_age

live['isfirst'] = live.birthord == 1
formula = 'totalwgt_lb ~ isfirst'
results = smf.ols(formula, data=live).fit()


live['agepreg2'] = live.agepreg**2
formula = 'totalwgt_lb ~ isfirst + agepreg + agepreg2'

live = live[live.prglngth>30]
resp = chap01soln.ReadFemResp()
resp.index = resp.caseid
join = live.join(resp, on='caseid', rsuffix='_r')



t = []
for name in join.columns:
    try:
        if join[name].var() < 1e-7:
            continue
        formula = 'totalwgt_lb ~ agepreg + ' + name
        model = smf.ols(formula, data=join)
        if model.nobs < len(join)/2:
            continue
        results = model.fit()
        except (ValueError, TypeError):
            continue

t.append((results.rsquared, name))


t.sort(reverse=True)
for mse, name in t[:30]:
print(name, mse)



formula = ('totalwgt_lb ~ agepreg + C(race) + babysex==1 + '
'nbrnaliv>1 + paydu==1 + totincr')
results = smf.ols(formula, data=join).fit()



live, firsts, others = first.MakeFrames()
df = live[live.prglngth>30]




import statsmodels.formula.api as smf
model = smf.logit('boy ~ agepreg', data=df)
results = model.fit()
SummarizeResults(results)


formula = 'boy ~ agepreg + hpagelb + birthord + C(race)'
model = smf.logit(formula, data=df)
results = model.fit()

actual = endog['boy']
baseline = actual.mean()



predict = (results.predict() >= 0.5)
true_pos = predict * actual
true_neg = (1 - predict) * (1 - actual)


acc = (sum(true_pos) + sum(true_neg)) / len(actual)


columns = ['agepreg', 'hpagelb', 'birthord', 'race']
new = pandas.DataFrame([[35, 39, 3, 2]], columns=columns)
y = results.predict(new)


def GroupByQualityAndDay(transactions):
    groups = transactions.groupby('quality')
    dailies = {}
    for name, group in groups:
        dailies[name] = GroupByDay(group)
            return dailies




def GroupByDay(transactions, func=np.mean):
    grouped = transactions[['date', 'ppg']].groupby('date')
    daily = grouped.aggregate(func)
    daily['date'] = daily.index
    start = daily.date[0]
    one_year = np.timedelta64(1, 'Y')
    daily['years'] = (daily.date - start) / one_year
        return daily


thinkplot.PrePlot(rows=3)
for i, (name, daily) in enumerate(dailies.items()):
    thinkplot.SubPlot(i+1)
    title = 'price per gram ($)' if i==0 else ''
    thinkplot.Config(ylim=[0, 20], title=title)
    thinkplot.Scatter(daily.index, daily.ppg, s=10, label=name)
    if i == 2:
        pyplot.xticks(rotation=30)
        else:
            thinkplot.Config(xticks=[])



def RunLinearModel(daily):
    model = smf.ols('ppg ~ years', data=daily)
    results = model.fit()
        return model, results



for name, daily in dailies.items():
    model, results = RunLinearModel(daily)
    print(name)
    regression.SummarizeResults(results)


def PlotFittedValues(model, results, label=''):
    years = model.exog[:,1]
    values = model.endog
    thinkplot.Scatter(years, values, s=15, label=label)
    thinkplot.Plot(years, results.fittedvalues, label='model')



dates = pandas.date_range(daily.index.min(), daily.index.max())
reindexed = daily.reindex(dates)



roll_mean = pandas.rolling_mean(reindexed.ppg, 30)
thinkplot.Plot(roll_mean.index, roll_mean)





ewma = pandas.ewma(reindexed.ppg, span=30)
thinkplot.Plot(ewma.index, ewma)



reindexed.ppg.fillna(ewma, inplace=True)

resid = (reindexed.ppg - ewma).dropna()
fake_data = ewma + thinkstats2.Resample(resid, len(reindexed))
reindexed.ppg.fillna(fake_data, inplace=True)


def SerialCorr(series, lag=1):
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    corr = thinkstats2.Corr(xs, ys)
        return corr


ewma = pandas.ewma(reindexed.ppg, span=30)
resid = reindexed.ppg - ewma
corr = SerialCorr(resid, 1)


import statsmodels.tsa.stattools as smtsa
acf = smtsa.acf(filled.resid, nlags=365, unbiased=True)




def AddWeeklySeasonality(daily):
    frisat = (daily.index.dayofweek==4) | (daily.index.dayofweek==5)
    fake = daily.copy()
    fake.ppg[frisat] += np.random.uniform(0, 2, frisat.sum())
        return fake




def GenerateSimplePrediction(results, years):
    n = len(years)
    inter = np.ones(n)
    d = dict(Intercept=inter, years=years)
    predict_df = pandas.DataFrame(d)
    predict = results.predict(predict_df)
        return predict

def SimulateResults(daily, iters=101):
    model, results = RunLinearModel(daily)
    fake = daily.copy()
    result_seq = []
    for i in range(iters):
        fake.ppg = results.fittedvalues + Resample(results.resid)
        _, fake_results = RunLinearModel(fake)
        result_seq.append(fake_results)
            return result_seq



def GeneratePredictions(result_seq, years, add_resid=False):
    n = len(years)
    d = dict(Intercept=np.ones(n), years=years, years2=years**2)
    predict_df = pandas.DataFrame(d)
    predict_seq = []
    for fake_results in result_seq:
        predict = fake_results.predict(predict_df)
        if add_resid:
            predict += thinkstats2.Resample(fake_results.resid, n)
            predict_seq.append(predict)
                return predict_seq


def PlotPredictions(daily, years, iters=101, percent=90):
    result_seq = SimulateResults(daily, iters=iters)
    p = (100 - percent) / 2
    percents = p, 100-p
    predict_seq = GeneratePredictions(result_seq, years, True)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.3, color='gray')
    predict_seq = GeneratePredictions(result_seq, years, False)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.5, color='gray')
    preg = nsfg.ReadFemPreg()
    complete = preg.query('outcome in [1, 3, 4]').prglngth
    cdf = thinkstats2.Cdf(complete, label='cdf')

class SurvivalFunction(object):
    def __init__(self, cdf, label=''):
        self.cdf = cdf
        self.label = label or cdf.label
            @property


def ts(self):
    return self.cdf.xs
    @property

def ss(self):
    return 1 - self.cdf.ps

sf = SurvivalFunction(cdf)


# class SurvivalFunction
def __getitem__(self, t):
    return self.Prob(t)

def Prob(self, t):
    return 1 - self.cdf.Prob(t)


thinkplot.Plot(sf)



# class SurvivalFunction
def MakeHazard(self, label=''):
    ss = self.ss
    lams = {}
    for i, t in enumerate(self.ts[:-1]):
        hazard = (ss[i] - ss[i+1]) / ss[i]
        lams[t] = hazard
        return HazardFunction(lams, label=label)


def __init__(self, d, label=''):
    self.series = pandas.Series(d)
    self.label = label





def EstimateHazardFunction(complete, ongoing, label=''):
    hist_complete = Counter(complete)
    hist_ongoing = Counter(ongoing)
    ts = list(hist_complete | hist_ongoing)
    ts.sort()
    at_risk = len(complete) + len(ongoing)
    lams = pandas.Series(index=ts)
    for t in ts:
        ended = hist_complete[t]
        censored = hist_ongoing[t]
        lams[t] = ended / at_risk
        at_risk -= ended + censored
        return HazardFunction(lams, label=label)




resp = chap01soln.ReadFemResp()
resp.cmmarrhx.replace([9997, 9998, 9999], np.nan, inplace=True)
resp['agemarry'] = (resp.cmmarrhx - resp.cmbirth) / 12.0
resp['age'] = (resp.cmintvw - resp.cmbirth) / 12.0
resp['agemarry'] = (resp.cmmarrhx - resp.cmbirth) / 12.0
resp['age'] = (resp.cmintvw - resp.cmbirth) / 12.0



complete = resp[resp.evrmarry==1].agemarry
ongoing = resp[resp.evrmarry==0].age


hf = EstimateHazardFunction(complete, ongoing)



# class HazardFunction:
def MakeSurvival(self):
    ts = self.series.index
    ss = (1 - self.series).cumprod()
    cdf = thinkstats2.Cdf(ts, 1-ss)
    sf = SurvivalFunction(cdf)
    return sf

def ResampleSurvival(resp, iters=101):
    low, high = resp.agemarry.min(), resp.agemarry.max()
    ts = np.arange(low, high, 1/12.0)
    ss_seq = []
    for i in range(iters):
        sample = thinkstats2.ResampleRowsWeighted(resp)
        hf, sf = EstimateSurvival(sample)
        ss_seq.append(sf.Probs(ts))
        low, high = thinkstats2.PercentileRows(ss_seq, [5, 95])
        thinkplot.FillBetween(ts, low, high)

resp5 = ReadFemResp1995()
resp6 = ReadFemResp2002()
resp7 = ReadFemResp2010()
resps = [resp5, resp6, resp7]


month0 = pandas.to_datetime('1899-12-15')
dates = [month0 + pandas.DateOffset(months=cm)
for cm in resp.cmbirth]
resp['decade'] = (pandas.DatetimeIndex(dates).year - 1900) // 10



for i in range(iters):
samples = [thinkstats2.ResampleRowsWeighted(resp)
for resp in resps]
sample = pandas.concat(samples, ignore_index=True)
groups = sample.groupby('decade')
EstimateSurvivalByDecade(groups, alpha=0.2)


EstimateSurvivalByDecade plots survival curves for each cohort:
def EstimateSurvivalByDecade(resp):
for name, group in groups:
hf, sf = EstimateSurvival(group)
thinkplot.Plot(sf)


# class HazardFunction
def Extend(self, other):
last = self.series.index[-1]
more = other.series[other.series.index > last]
self.series = pandas.concat([self.series, more])

def PlotPredictionsByDecade(groups):
hfs = []
for name, group in groups:
hf, sf = EstimateSurvival(group)
hfs.append(hf)
thinkplot.PrePlot(len(hfs))
for i, hf in enumerate(hfs):
if i > 0:
hf.Extend(hfs[i-1])
sf = hf.MakeSurvival()
thinkplot.Plot(sf)

# class SurvivalFunction
def MakePmf(self, filler=None):
pmf = thinkstats2.Pmf()
for val, prob in self.cdf.Items():
pmf.Set(val, prob)
cutoff = self.cdf.ps[-1]
if filler is not None:
pmf[filler] = 1-cutoff
    return pmf




# class SurvivalFunction
def RemainingLifetime(self, filler=None, func=thinkstats2.Pmf.Mean):
pmf = self.MakePmf(filler=filler)
d = {}
for t in sorted(pmf.Values())[:-1]:
pmf[t] = 0
pmf.Normalize()
d[t] = func(pmf) - t
    return pandas.Series(d)



rem_life1 = sf1.RemainingLifetime()
thinkplot.Plot(rem_life1)
func = lambda pmf: pmf.Percentile(50)
rem_life2 = sf2.RemainingLifetime(filler=np.inf, func=func)
thinkplot.Plot(rem_life2)



class Normal(object):
def __init__(self, mu, sigma2):
self.mu = mu
self.sigma2 = sigma2
def __str__(self):
    return 'N(%g, %g)' % (self.mu, self.sigma2)


def Sum(self, n):
    return Normal(n * self.mu, n * self.sigma2)


def __mul__(self, factor):
    return Normal(factor * self.mu, factor**2 * self.sigma2)
def __div__(self, divisor):
    return 1 / divisor * self


def MakeExpoSamples(beta=2.0, iters=1000):
samples = []
for n in [1, 10, 100]:
sample = [np.sum(np.random.exponential(beta, n))
for _ in range(iters)]
samples.append((n, sample))
    return samples

def NormalPlotSamples(samples, plot=1, ylabel=''):
    for n, sample in samples:
        thinkplot.SubPlot(plot)
        thinkstats2.NormalProbabilityPlot(sample)
        thinkplot.Config(title='n=%d' % n, ylabel=ylabel)
        plot += 1

def GenerateCorrelated(rho, n):
    x = random.gauss(0, 1)
    yield x
    sigma = math.sqrt(1 - rho**2)
    for _ in range(n-1):
        x = random.gauss(x*rho, sigma)
        yield x



def GenerateExpoCorrelated(rho, n):
normal = list(GenerateCorrelated(rho, n))
uniform = scipy.stats.norm.cdf(normal)
expo = scipy.stats.expon.ppf(uniform)
    return expo





dist1 = SamplingDistMean(live.prglngth, len(firsts))
dist2 = SamplingDistMean(live.prglngth, len(others))


def SamplingDistMean(data, n):
mean, var = data.mean(), data.var()
dist = Normal(mean, var)
    return dist.Sum(n) / n


def __sub__(self, other):
    return Normal(self.mu - other.mu,
    
self.sigma2 + other.sigma2)



def StudentCdf(n):
ts = np.linspace(-3, 3, 101)
ps = scipy.stats.t.cdf(ts, df=n-2)
rs = ts / np.sqrt(n - 2 + ts**2)
    return thinkstats2.Cdf(rs, ps)





t = r * math.sqrt((n-2) / (1-r**2))
p_value = 1 - scipy.stats.t.cdf(t, df=n-2)

def ChiSquaredCdf(n):
xs = np.linspace(0, 25, 101)
ps = scipy.stats.chi2.cdf(xs, df=n-1)
    return thinkstats2.Cdf(xs, ps)

p_value = 1 - scipy.stats.chi2.cdf(chi2, df=n-1)


