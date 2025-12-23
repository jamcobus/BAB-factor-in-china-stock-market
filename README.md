# A股市场的BAB策略有效研究
简单的BAB策略
如果你跑了，那你会得到：
<img width="806" height="403" alt="image" src="https://github.com/user-attachments/assets/f00b8eb7-0d60-4b9a-8c49-f6ba35d6769d" />

图1展示了1996年至2024年中国A股市场波动率，初始值由于滚动回归时间窗口导致极端值，波动率门槛值为所有BAB因子滚动窗口波动率的中值。
<img width="872" height="436" alt="image" src="https://github.com/user-attachments/assets/7c68dc7f-226e-4edc-a4d4-5c8c80aac866" />

图2展示了基于波动率中值划分的BAB策略回报累积和，由于滚动回归时间窗口导致初始的极端值，高波动BAB时期与低波动BAB时期间断点自动连接，便于趋势观察。
<img width="865" height="649" alt="image" src="https://github.com/user-attachments/assets/61748358-a381-4243-be3e-3887de041ca6" />

图3 展示了BAB策略的全部回报，在过去三十年里运行的BAB策略实现回报为负，这可以解释为观察期内的A股市场并没有发生温和的市场衰退，按照Xia Xu(2025)的解释，积极的市场会对BAB策略带来损失，由此可见A股市场在2015年以前是增长的。
 Xia Xu(2025) Market neutrality and beta crashes. Journal of Empirical Finance, 80,101577
 <img width="498" height="374" alt="image" src="https://github.com/user-attachments/assets/b1f8c02c-bf24-436b-b3a3-8373fdd50bec" />
<img width="518" height="389" alt="image" src="https://github.com/user-attachments/assets/199c6fe7-c7f5-4f32-8a98-95d66eb376b4" />

图4 和图5展示了BAC和BAV因子随着组合相关性和市场波动率上升，策略回报的变化趋势。可以看到BAC和BAV策略回报基本不随市场相关因素的变化，图像无明显倾斜。表明在市场相关因素变动时，BAC和BAV策略可以成为风险规避策略
<img width="864" height="518" alt="image" src="https://github.com/user-attachments/assets/b29ed1f0-cda9-480d-b414-b607d5e0eefd" />

图6 展示了Low-High波动率时期FF5因子模型各相关系数的比较，首先截距项α极小，分别为-0.0134和0.0021，且不显著。无论高低波动率时期，FF5模型都很好解释了BAB策略回报。
