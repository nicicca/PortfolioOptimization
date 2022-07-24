# Import Packages
import time
import numpy as np
import pandas as pd
import yfinance as yf
from statistics import mean
from scipy.optimize import minimize, LinearConstraint


class PortfolioOptimization:
    def __init__(self, StockMnemonics: list, period: list, allow_for_short=False, tc=0.01,
                 rebalance=False, rebalance_capital=None, rebalance_weights = None):
        print("-"*400, "\n New portfolio optimization has started for period", " until ".join(period),
              "\n", "Start downloading data.")
        # Initialize the time counting
        start = time.time()

        # Import data with the specified mnemonics
        InfoSet = ImportData(StockMnemonics, period)

        # Return classified DataSet
        ClassifiedInfoSet = StockRanking(InfoSet.returns)

        # Return the final weights
        self.Results = Maximization(ClassifiedInfoSet.share, InfoSet.returns, allow_for_short=allow_for_short, tc=tc)

        # Return the processing time
        stop = time.time()
        print("-" * 400, "\n With the current algorithm for the financial products:\n", ", ".join(StockMnemonics),
              "\n the portfolio optimization was implemented in seconds:", np.round(stop - start, 2), ".\n", "-" * 400)

        if rebalance:
            # Define weights
            self.FinalWeights = np.round(self.Results.result_max.x * rebalance_capital, 4)
            diff = np.round(self.FinalWeights - rebalance_weights, 4)
            val = [str(x) for x in diff.tolist()]

            # Store amount of capital
            self.capital = float(rebalance_capital)

            # Print results
            if allow_for_short:
                print(" With leverage, you should add/subtract to the previous portfolio weights respectively:\n", ", ".join(val),
                      " euros.\n", "-" * 400)
            else:
                print(" With no leverage, you should add/subtract to the previous portfolio weights respectively:\n", ", ".join(val),
                      " euros.\n", "-" * 400)

        else:

            InputString = input(" To maximize the excessive returns with the current set of financial products, "
                                "specify the initial capital:")

            # Store the Input string
            self.capital = float(InputString)

            # Compute shares
            if type(float(InputString)) == float:
                self.FinalWeights = np.round(self.Results.result_max.x * float(InputString), 4)
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
    def __init__(self, shares: dict, returns: pd.DataFrame, allow_for_short=True, tc=0.01):
        # Define initial conditions for maximization
        weights = np.array([[0.1] * (returns.shape[1] - 1)]).transpose()
        self.maximization_problem(weights, returns, shares, allow_for_short=allow_for_short, tc=tc)

    def maximization_problem(self, weights: np.array, returns: pd.DataFrame, shares: dict, allow_for_short=True,
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
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, shares, allow_for_short),
                                       method='SLSQP',
                                       constraints=C)
        else:
            bounds = [(0, 1)] * weights.shape[0]
            self.result_max = minimize(fun=self.maximand_function, x0=weights, args=(returns, shares, tc),
                                       method='SLSQP', bounds=bounds,
                                       constraints=C)

    @staticmethod
    def maximand_function(weights: np.array, returns: pd.DataFrame, shares: dict, tc=0.01):

        # Define the excess returns with respect to the risk-free
        diff = returns.to_numpy()[:, :-1] - returns.to_numpy()[:, -1].reshape([returns.to_numpy()[:, -1].shape[0], 1])

        # Define the excess negative returns
        diff_min = returns[returns < 0].iloc[:, :-1].to_numpy() - returns.iloc[:, -1].to_numpy().reshape(
            [returns.to_numpy()[:, -1].shape[0], 1])

        # Extract the variance-covariance matrix
        var_cov = np.ma.cov(np.ma.masked_invalid(diff_min.transpose()))

        # Expand dimension of weights matrix
        weights = weights[:, np.newaxis]

        # Define the maximand function as a linear combination Sortino Ratio with t-Student Nuisances
        arg = - (1 - tc) * diff.mean(axis=0).reshape(
            [returns.iloc[:, :-1].shape[1], 1]).transpose() @ weights @ np.linalg.inv(
            weights.transpose() @ np.multiply(
                mean(shares['dof']) / (mean(shares['dof']) - 2),
                np.cov(np.ma.getdata(var_cov))) @ weights)
        return arg


class ImportData:
    def __init__(self, StockMnemonics: list, period: list):
        # Add the risk-free interest rate mnemonic symbol
        RiskFreeMnemonics = ['^TNX']  # Treasury Yield 10 Years
        StockMnemonics = StockMnemonics + RiskFreeMnemonics

        # Download the mnemonics specified by the user
        dataframe = yf.download(" ".join(StockMnemonics), start=period[0], end=period[1])
        self.returns = dataframe.iloc[:, :len(StockMnemonics) - 1].diff().dropna().assign(
            RiskFree=dataframe.iloc[1:, len(StockMnemonics) - 1])


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

        # Extract parameters describing the centered t-student distribution
        # Compute observed variance
        varr = returns.var().round(3)

        # Observed degrees of freedom
        dff = -2 * varr / (1 - varr)

        # Classify
        if dff > 30:
            return ["Normal", dff]
        else:
            return ["Student-t", dff]
