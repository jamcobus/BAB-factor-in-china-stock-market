# 这是一个示例 Python 脚本。

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 保证pycharm环境可以正常画图

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    ##################数据准备与因子构建#########################

    # 读取数据
    # 股票收益率数据（假设包含股票代码、日期、收益率）
    stock_returns = pd.read_csv("TRD_Mnth.csv", parse_dates=["date"])

    # FF5因子数据（假设包含日期、MKT、SMB、HML、RMW、CMA）
    ff5_factors = pd.read_csv("fivefactor_monthly1.csv", parse_dates=["date"])


    # 计算每只股票的Beta值（以过去12个月数据滚动计算）
    def calculate_beta(stock_returns, market_returns,rf, window=12):
        stock_returns = stock_returns.set_index(["date"])
        market_returns = market_returns.set_index("date")
        r_f = rf.set_index("date")

        # 合并股票收益率与市场收益率
        merged_data = stock_returns.join(market_returns, on="date", how="inner").join(r_f,on="date",how="inner")
        merged_data["exreturn"] = merged_data["ret"]-merged_data["rf"]
        merged_data['MKT_lag1'] = merged_data.groupby('stock_id')['MKT'].shift(1)
        merged_data = merged_data.dropna(subset=["MKT_lag1"])

        # 计算Beta
        def rolling_beta(group,_add_constant=True):
            try:
                a = group["MKT"]
                b = group["MKT_lag1"]
                c = group["exreturn"]
                X = pd.concat([a,b], axis=1)
                if _add_constant:
                    X = add_constant(X) 
                    model = RollingOLS(endog=c, exog=X, window=window)
                    results = model.fit()
                    params = results.params
                    group["beta"] = list(map(lambda x, y: x + y,params.iloc[:,1],params.iloc[:,2] ))
                    #group["beta"]= group["ret"].rolling(window=window).cov(group["MKT"]) / group["MKT"].rolling(window=window).var()
                return group
            except Exception as e:
                print(f"Error: {e}")
                return None

        merged_data = merged_data.groupby("stock_id").apply(rolling_beta)
        merged_data = merged_data.drop(columns=['stock_id']) #要去掉stock_id列不然会冲突
        merged_data = merged_data.reset_index()
        # merged_data = merged_data.set_index(["stock_id"])
        return merged_data


    # 计算Beta
    stock_returns_with_beta = calculate_beta(stock_returns, ff5_factors[["date", "MKT"]],ff5_factors[["date","rf"]])
    # print(stock_returns_with_beta.head())
    # 计算每个月的BAB因子
    def build_bab_factor(stock_returns_with_beta):
        # 删除 Beta 值为 NaN 的行（因为是以前n个月窗口期计算的beta，所以每个股票的前n-1个月值都是NaN，这里删去）
        stock_returns_with_beta = stock_returns_with_beta.dropna(subset=["beta"])

        # 按 Beta 排序分组（每月末调整）
        def safe_qcut(x, q=5):
            try:
                return pd.qcut(x, q=q, labels=False, duplicates="drop")
            except ValueError:
                return pd.Series([np.nan] * len(x), index=x.index)

        stock_returns_with_beta["beta_rank"] = stock_returns_with_beta.groupby("date")["beta"].transform(safe_qcut)

        # 删除分组失败的日期
        stock_returns_with_beta = stock_returns_with_beta.dropna(subset=["beta_rank"])

        # 构建 BAB 因子
        bab_portfolio = stock_returns_with_beta.groupby("date").apply(
            lambda x: pd.Series({
                "bab_return": x.loc[x["beta_rank"] == 0, "exreturn"].mean() - x.loc[x["beta_rank"] == 4, "exreturn"].mean()
            })
        ).reset_index()
             

        return bab_portfolio
    #ma_bab_por,后面再补

    # 构建BAB因子
    bab_portfolio = build_bab_factor(stock_returns_with_beta)


    ###############################BAB分解为BAC和BAV，并且研究和市场相关性和波动性的关系##############################

    def decompose_bab(stock_returns_with_beta):
        # 删除 Beta 值为 NaN 的行
        stock_returns_with_beta = stock_returns_with_beta.dropna(subset=["beta"])

        # 计算每只股票与市场的相关性（rho）和波动率（sigma）
        stock_stats = stock_returns_with_beta.groupby("stock_id").apply(
            lambda x: pd.Series({
                "rho": x["ret"].corr(x["MKT"]),
                "sigma": x["ret"].std()
            })
        ).reset_index()


        # 在考虑价值投资时，应更关注组合的长期趋势而不是短期波动，因此只考虑全期相关性和波动性，而不是滚动相关性和波动性
        # # 设置滚动窗口大小（如 20 天）
        # window = 12
        #
        # # 计算每只股票与市场的滚动相关性
        # stock_stats["rolling_rho"] = stock_returns_with_beta.groupby("stock_id").apply(
        #     lambda x: x["exreturn"].rolling(window=window).corr(x["MKT"])
        # ).reset_index(level=0, drop=True)
        #
        # # 计算每只股票的滚动波动率
        # stock_stats["rolling_sigma"] = stock_returns_with_beta.groupby("stock_id")["exreturn"].transform(
        #     lambda x: x.rolling(window=window).std()
        # )

        # 合并数据
        merged_data = stock_returns_with_beta.merge(stock_stats, on="stock_id")

        # 构建BAC和BAV
        def build_bac_bav(group):
            group["rho_rank"] = pd.qcut(group["rho"], 5, labels=False)  # 按相关性分组
            group["sigma_rank"] = pd.qcut(group["sigma"], 5, labels=False)  # 按波动率分组
            return group

        merged_data = merged_data.groupby("date").apply(build_bac_bav)

        merged_data = merged_data.drop(columns=['date']) # 要去掉date列不然会冲突

        # 计算BAC和BAV
        bac_bav_portfolio = merged_data.groupby("date").apply(
            lambda x: pd.Series({
                "bac_return": x.loc[x["rho_rank"] == 0, "ret"].mean() - x.loc[x["rho_rank"] == 4, "ret"].mean(),
                "bav_return": x.loc[x["sigma_rank"] == 0, "ret"].mean() - x.loc[x["sigma_rank"] == 4, "ret"].mean()
            })
        ).reset_index()

        # 计算市场平均相关性
        market_correlation = merged_data.groupby("date")["rho"].mean().reset_index()
        market_correlation.columns = ["date", "market_rho"]

        # 计算市场平均波动率
        market_volatility = merged_data.groupby("date")["sigma"].mean().reset_index()
        market_volatility.columns = ["date", "market_sigma"]

        # 合并 BAC 收益与市场相关性
        bac_correlation = bac_bav_portfolio[["date", "bac_return"]].merge(market_correlation, on="date", how="inner")

        # 合并 BAV 收益与市场波动率
        bav_volatility = bac_bav_portfolio[["date", "bav_return"]].merge(market_volatility, on="date", how="inner")

        # 计算 BAC 收益与市场相关性的相关性
        bac_rho_corr = bac_correlation["bac_return"].corr(bac_correlation["market_rho"])
        print("BAC 收益与市场相关性的相关性：", bac_rho_corr)

        # 计算 BAV 收益与市场波动率的相关性
        bav_sigma_corr = bav_volatility["bav_return"].corr(bav_volatility["market_sigma"])
        print("BAV 收益与市场波动率的相关性：", bav_sigma_corr)

        # 其实上面得到BAC，BAV收益和市场相关性波动率结果并不是特别好(与市场相关性的相关性是负值，但是接近0；与市场波动性相关性是正值），猜测可能是市场相关性平均波动率的计算方式有优化空间

        # 绘制 BAC 与市场相关性散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(bac_correlation["market_rho"], bac_correlation["bac_return"])
        plt.title("BAC Return vs Market Correlation")
        plt.xlabel("Market Correlation (rho)")
        plt.ylabel("BAC Return")
        plt.grid(True)
        plt.show()

        # 绘制 BAV 与市场波动率散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(bav_volatility["market_sigma"], bav_volatility["bav_return"])
        plt.title("BAV Return vs Market Volatility")
        plt.xlabel("Market Volatility (sigma)")
        plt.ylabel("BAV Return")
        plt.grid(True)
        plt.show()


        return bac_bav_portfolio

    # 分解BAB为BAC和BAV
    bac_bav_portfolio = decompose_bab(stock_returns_with_beta)


    ##########################时序回归分析（FF3/FF5模型）#################################

    def run_regression(bab_portfolio, ff5_factors, name):
        # 合并BAB、BAC、BAN收益与FF5因子
        merged_data = bab_portfolio.merge(ff5_factors, on="date")

                                           
        # FF3模型回归
        ff3_model = sm.OLS(merged_data[name], sm.add_constant(merged_data[["MKT", "SMB", "HML"]])).fit()
        print("FF3 Model Results:")
        print(ff3_model.summary())

        # FF5模型回归
        ff5_model = sm.OLS(merged_data[name],
                           sm.add_constant(merged_data[["MKT", "SMB", "HML", "RMW", "CMA"]])).fit()
        print("FF5 Model Results:")
        print(ff5_model.summary())

        return ff3_model, ff5_model


    # 对BAB因子进行多元线性回归
    ff3_model, ff5_model = run_regression(bab_portfolio, ff5_factors, "bab_return")
    # BAB因子通过做多低beta资产、做空高beta资产构建，可能与FF3中的市场因子（MKT）或规模因子（SMB）高度相关，解释了BAB因子的主要收益来源，导致截距（alpha）不显著

    # 对BAC因子进行多元线性回归
    ff3_model_BAC, ff5_model_BAC = run_regression(bac_bav_portfolio, ff5_factors, "bac_return")

    # 对BAV因子进行多元线性回归
    ff3_model_BAV, ff5_model_BAV = run_regression(bac_bav_portfolio, ff5_factors, "bav_return")

    #####################波动率分时期分析###############################

    # 计算市场平均收益率（所有股票收益率的等权平均）
    #market_returns = pd.read_csv("TRD_Cnmont.csv", parse_dates=["Trdmnt"])
    market_returns = stock_returns.groupby("date")["ret"].mean().reset_index()
    market_returns.columns = ["date", "market_return"]

    # 这里要对波动率分时期分析，所以与上面关注长期波动率不同，这里就反而应该关注滚动期内的波动率
    # 设置滚动窗口大小（这里和上面计算beta保持一致）
    window = 12

    # 计算滚动波动率
    market_returns["market_vol"] = market_returns["market_return"].rolling(window=window).std()

    # 合并 BAB 因子数据和市场波动率数据
    bab_portfolio = bab_portfolio.merge(market_returns[["date", "market_vol"]], on="date", how="left")

    # 划分高波动期和低波动期（以中位数为阈值）
    vol_threshold = bab_portfolio["market_vol"].median()
    bab_portfolio["vol_regime"] = np.where(bab_portfolio["market_vol"] > vol_threshold, "High", "Low")

    # 绘制市场波动率
    plt.figure(figsize=(12, 6))
    plt.plot(bab_portfolio["date"], bab_portfolio["market_vol"], label="Market Volatility")
    plt.axhline(vol_threshold, color="red", linestyle="--", label="Volatility Threshold")
    plt.title("Market Volatility Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()
    # 这里市场波动率是滚动窗口12个月去计算并绘制的，因此在曲线开头，数据量不足，导致计算结果不稳定，波动率可能被高估

    # 分析 BAB 因子在不同波动时期的表现
    high_vol_period = bab_portfolio[bab_portfolio["vol_regime"] == "High"]
    low_vol_period = bab_portfolio[bab_portfolio["vol_regime"] == "Low"]

    print("高波动期 BAB 因子平均收益率：", high_vol_period["bab_return"].mean())
    print("低波动期 BAB 因子平均收益率：", low_vol_period["bab_return"].mean())
    # 高-0.023，低-0.0026，低波动时期收益率更好

    # 绘制 BAB 因子在不同时期的表现
    plt.figure(figsize=(12, 6))
    plt.plot(high_vol_period["date"], high_vol_period["bab_return"].cumsum(), label="High Volatility Period")
    plt.plot(low_vol_period["date"], low_vol_period["bab_return"].cumsum(), label="Low Volatility Period")
    plt.title("BAB Strategy Performance in Different Volatility Regimes")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()
    # 可以明显看到，低波动时期的BAB因子显著高于高波动时期的BAB因子

    # 低波动期与高波动期下BAB的FF3，FF5回归结果
    ff3_model_high, ff5_model_high = run_regression(high_vol_period, ff5_factors, "bab_return")
    ff3_model_low, ff5_model_low = run_regression(low_vol_period, ff5_factors, "bab_return")

    # 提取回归系数，这里就先只看FF5了
    low_coef = ff5_model_low.params  # 低波动期系数
    high_coef = ff5_model_high.params  # 高波动期系数

    # 绘制柱状图，对比高低时期的回归值
    plt.figure(figsize=(10, 6))
    x = range(len(low_coef))
    plt.bar(x, low_coef, width=0.4, label="Low Volatility", align="center")
    plt.bar(x, high_coef, width=0.4, label="High Volatility", align="edge")
    plt.xticks(x, low_coef.index)
    plt.xlabel("FF5 Factors")
    plt.ylabel("Coefficient")
    plt.title("FF5 Regression Coefficients: Low vs High Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()
    # BAB因子通过做多低beta资产、做空高beta资产构建。低beta资产对市场波动的敏感性较低，而高beta资产对市场波动的敏感性较高。
    # 当市场因子（MKT）上升时：
    # 高beta资产价格上涨更多，做空高beta资产会亏损。
    # 低beta资产价格上涨较少，做多低beta资产收益有限。
    # 因此，BAB因子在市场上涨时表现较差，与市场因子呈负相关。
    # 上面图像看出，低波动时期BAB因子，对市场因子（MKT)的暴露程度更高；而高波动时期的BAB因子，对盈利能力因子(RMW)的暴露程度更高


    # 计算夏普率
    def calculate_sharpe(returns):
        return returns.mean() / returns.std()

    # 计算低波动期和高波动期的夏普率
    sharpe_low = calculate_sharpe(low_vol_period["bab_return"])
    sharpe_high = calculate_sharpe(high_vol_period["bab_return"])
    print("低波动期 BAB 夏普率：", sharpe_low)
    print("高波动期 BAB 夏普率：", sharpe_high)
    #  低波动期BAB夏普率更高

    # 计算低波动期和高波动期的BAB和RMW、CMA的相关性
    high_vol_period = high_vol_period.merge(ff5_factors, on="date", how="inner")
    low_vol_period = low_vol_period.merge(ff5_factors, on="date", how="inner")
    corr_rmw_low = low_vol_period["bab_return"].corr(low_vol_period["RMW"])
    corr_cma_low = low_vol_period["bab_return"].corr(low_vol_period["CMA"])
    corr_rmw_high = high_vol_period["bab_return"].corr(high_vol_period["RMW"])
    corr_cma_high = high_vol_period["bab_return"].corr(high_vol_period["CMA"])
    print("低波动期 BAB 与 RMW 的相关性：", corr_rmw_low)
    print("低波动期 BAB 与 CMA 的相关性：", corr_cma_low)
    print("高波动期 BAB 与 RMW 的相关性：", corr_rmw_high)
    print("高波动期 BAB 与 CMA 的相关性：", corr_cma_high)
    #  低波动期BAB与RMW、CMA相关性弱。

    #####################BAB因子可视化##################

    # 确保索引是时间序列
    if not isinstance(bab_portfolio.index, pd.DatetimeIndex):
        bab_portfolio = bab_portfolio.set_index("date")
        bab_portfolio.index = pd.to_datetime(bab_portfolio.index)

    # 计算累计收益并绘图，这里使用cumsum()显示的是收益率的累计值，从而更直观展示策略的长期表现
    bab_portfolio.cumsum().plot(y="bab_return", title="BAB Strategy Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.show()
    # 市场累计收益为正，在市场长期上涨时：
    # 高beta资产表现通常优于低beta资产，因为高beta资产对市场上涨更敏感,
    # 而BAB因子做多低beta资产，做kong高beta资产，低beta资产的收益无法完全抵消这部分亏损，导致BAB因子的累计收益下降

