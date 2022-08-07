# Import packages
import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

# Import global statement
from scipy.stats import t
from scipy.stats.mstats import gmean
from scipy.optimize import minimize, LinearConstraint

# Import R packages
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import *

# Set up enviroment for interaction between Python and R
rpy2.robjects.numpy2ri.activate()
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# # Install packages
r.require('fitHeavyTail')

# Load packages
Heavy = importr('fitHeavyTail')
warnings.filterwarnings("ignore")


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
############################################# Sortino Index Maximization ##############################################

class PortfolioOptimization:
    def __init__(self, StockMnemonics: list, period: list, allow_for_short=False, tc=0.01,
                 rebalance=False, capital=None, rebalance_weights=None):
        print("-" * 400, "\n New portfolio optimization has started for period", " until ".join(period),
              "\n", "Start downloading data.")
        # Initialize the time counting
        start = time.time()

        # Import data with the specified mnemonics
        InfoSet = ImportData(StockMnemonics, period)

        # Return classified DataSet
        ClassifiedInfoSet = StockRanking(InfoSet.returns)

        if rebalance:

            # Define weights
            weights = np.divide(rebalance_weights,float(capital)).reshape([rebalance_weights.shape[0], 1])

            # Return the final weights
            self.Results = Maximization(InfoSet.returns, weights, allow_for_short=allow_for_short, tc=tc)

            # Return the processing time
            stop = time.time()
            print("-" * 400, "\n With the current algorithm for the financial products:\n", ", ".join(StockMnemonics),
                  "\n the portfolio optimization was implemented in seconds:", np.round(stop - start, 2), ".\n",
                  "-" * 400)

            # Define weights
            self.FinalWeights = np.round(self.Results.result_max.x * capital, 4)
            self.diff = np.round(self.FinalWeights - rebalance_weights, 2)
            val = [str(x) for x in self.diff.tolist()]

            # Store amount of capital
            self.capital = float(capital)

            # Print results
            if allow_for_short:
                print(" With leverage, you should add/subtract to the previous portfolio weights respectively:\n",
                      ", ".join(val),
                      " euros.\n", "-" * 400)
            else:
                print(" With no leverage, you should add/subtract to the previous portfolio weights respectively:\n",
                      ", ".join(val),
                      " euros.\n", "-" * 400)

        else:
            # Define weights
            weights = np.array([[int(len(StockMnemonics)) / (int(len(StockMnemonics)) * 100)]
                                * (InfoSet.returns.shape[1] - 1)]).transpose()

            # Return the final weights
            self.Results = Maximization(InfoSet.returns, weights, allow_for_short=allow_for_short, tc=tc)

            # Return the processing time
            stop = time.time()
            print("-" * 400, "\n With the current algorithm for the financial products:\n", ", ".join(StockMnemonics),
                  "\n the portfolio optimization was implemented in seconds:", np.round(stop - start, 2), ".\n",
                  "-" * 400)

            print("To maximize the excessive returns with the current set of financial products, "
                                "specify the initial capital.")

            InputString = str(capital)

            # Store the Input string
            self.capital = int(InputString)

            # Compute shares
            if type(float(InputString)) == float:
                self.FinalWeights = np.round(self.Results.result_max.x * float(InputString), 4)

                if all(self.FinalWeights / InfoSet.dataframe.iloc[-1, :len(StockMnemonics)].to_numpy()
                       < 1):
                    print("The initial capital does not allow to buy the full property of all the stocks.\n"
                          " Consider buying CFDs. ")
                elif any(self.FinalWeights / InfoSet.dataframe.iloc[-1, :len(StockMnemonics)].to_numpy()
                         < 1):
                    print("The initial capital does not allow to buy "
                          "the full property of stocks for all of them.\n Consider buying CFDs. ")

                # Change type per
                val = [str(x) for x in self.FinalWeights.tolist()]
            else:
                raise TypeError("Please specify the cipher without any comma or special characters.")

            # Print results
            if allow_for_short:
                print("-" * 400, "\n With leverage, you should invest this month respectively:\n", ", ".join(val),
                      " euros.")
            else:
                print("-" * 400, "\n With no leverage, you should invest this month respectively:\n", ", ".join(val),
                      " euros.")


class Maximization:
    def __init__(self, returns: pd.DataFrame, weights: np.array, allow_for_short=True, tc=0.01):
        # Define initial conditions for maximization
        self.result_max = None
        self.maximization_problem(weights, returns, allow_for_short=allow_for_short, tc=tc)

    def maximization_problem(self, weights: np.array, returns: pd.DataFrame, allow_for_short=True,
                             tc=0.01):

        # Define constraints
        # Matrix constraint
        A = np.ones([1, weights.shape[0]])

        # Upper and lower bound equal => equality constraint
        ub_lb = np.ones(1)

        # Call class linear constraint
        C = LinearConstraint(A, ub_lb, ub_lb)

        # Impose bound on weights depending on boolean: "allow_for_short".
        if allow_for_short:
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, allow_for_short),
                                       method='SLSQP',
                                       constraints=C)

        else:
            bounds = [(x[0], 0.9) for x in weights]
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, tc),
                                       method='SLSQP', bounds=bounds,
                                       constraints=C)

    @staticmethod
    def maximand_function(weights: np.array, returns: pd.DataFrame, tc=0.01):

        # Define the excess negative returns
        diff_min = (1 + returns[returns < 0]).cumprod().to_numpy()[:, :-1] - \
                   (1 + returns).cumprod().to_numpy()[:, -1].reshape([returns.to_numpy()[:, -1].shape[0], 1])

        # Extract the variance-covariance matrix
        var_cov = np.ma.cov(np.ma.masked_invalid(diff_min.transpose()))

        # Expand dimension of weights matrix
        weights = weights[:, np.newaxis]
        num = (1 + returns).cumprod().to_numpy()[-1, :-1] - (1 + returns).cumprod().to_numpy()[-1, -1]
        num = num.reshape([returns.iloc[:, :-1].shape[1], 1])

        # Extract the degrees of freedom of the multivariate distribution
        Mult_t = Heavy.fit_mvt(np.ma.masked_invalid(diff_min))
        Mult_t = dict(zip(Mult_t.names, map(list, list(Mult_t))))
        doff = Mult_t['nu'][0]

        # Define the maximand function as a linear combination Sortino Ratio with t-Student Nuisances
        arg = - (1 - tc) * num.transpose() @ weights @ np.linalg.inv(
            weights.transpose() @ np.sqrt(np.multiply(
                doff / (doff - 2),
                np.cov(np.ma.getdata(var_cov)))) @ weights)
        return arg


class ImportData:
    def __init__(self, StockMnemonics: list, period: list):
        # Add the risk-free interest rate mnemonic symbol
        RiskFreeMnemonics = ['^FVX']  # Treasury Yield 10 Years TNX
        # Treasury Yield 5 Years FVX

        StockMnemonics = StockMnemonics + RiskFreeMnemonics

        # Download the mnemonics specified by the user
        self.dataframe = yf.download(" ".join(StockMnemonics), start=period[0], end=period[1])
        self.returns = self.dataframe.iloc[:, :len(StockMnemonics) - 1].assign(
            RiskFree=self.dataframe.iloc[1:, len(StockMnemonics) - 1]).pct_change().dropna()
        self.returns.columns = StockMnemonics

        # Redefine the columns
        self.prices = self.dataframe.iloc[:, :len(StockMnemonics)]
        self.prices.columns = StockMnemonics


class StockRanking:
    def __init__(self, returns: pd.DataFrame):
        # Create a set of labels for each return
        label_column = []
        for i in range(returns.shape[1] - 1):
            temp = self.classify_distribution(returns.iloc[:, i].to_numpy())
            label_column.append(temp)

        # Extract the alpha and beta
        self.share = self.alpha_beta(label_column)

    @staticmethod
    def alpha_beta(label_column: list):

        # Count the elements
        alpha = sum([x.count("Normal") for x in label_column])
        beta = sum([x.count("Student-t") for x in label_column])

        return {'alpha': float(alpha / (alpha + beta)), 'beta': float(beta / (alpha + beta)),
                "dof": [x[1] for x in label_column]}

    @staticmethod
    def classify_distribution(returns: np.array):

        # Observed degrees of freedom
        dff, _, _ = t.fit(returns, loc=returns.mean(), scale=returns.var())

        # Classify
        if dff > 30:
            return ["Normal", dff]
        else:
            return ["Student-t", dff]
