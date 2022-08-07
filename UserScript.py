# Import packages you like the most on top of the class for PortofolioOptimization
import dateutil.relativedelta
from datetime import datetime
from Sortino_Index import PortfolioOptimization

# Set the date of today and obtain data from a lagged period
now = datetime.now()
dt_1 = now - dateutil.relativedelta.relativedelta(months=1)
dt = now.strftime("%Y-%m-%d")
dt_1 = dt_1.strftime("%Y-%m-%d")

# Take the list of symbols from the list of Stocks available on Yahoo Finance.
# For example, trending tickers:  https://finance.yahoo.com/lookup
ETF_list = ['^GSPC', "^FTSE", "^NDX", "^SP500-45", "NZAC"]

# Run the Portfolio Analysis by setting:
# 1. At least two stock symbols and date range in yy-mm-dd format
# 2. A boolean 'allow_for_shorting' is set to default to 'False'.
# 3. A value 'tc' between 0 and 1, as a fixed share of returns. It is set to default to 0.01.
OptimizedPortfolio = PortfolioOptimization(ETF_list, ['2005-01-01', dt_1], capital=2500)

# Extract the optimizing weights
q_1 = OptimizedPortfolio.FinalWeights

# NOTE: After running the file, specify the initial capital with plain ciphers
# - neither commas nor other special characters.
#
# To rebalance the portfolio, run again the Portfolio optimization at t=2,
# Activate the rebalance mode by setting:
# 1. The boolean "rebalance" to True. The default is False.
# 2. The "rebalance_capital" equal to the capital of the previous optimization
# 3. The "rebalance_weights" equal to the weights of the previous optimization
delta = 300 # Money you are adding time by time
new_capital = OptimizedPortfolio.capital + delta
OptimizedPortfolio = PortfolioOptimization(ETF_list, ['2005-01-01', dt],
                                           rebalance=True, capital=new_capital,
                                           rebalance_weights=OptimizedPortfolio.FinalWeights)

