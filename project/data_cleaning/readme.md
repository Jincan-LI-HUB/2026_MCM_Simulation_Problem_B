# 清洗后数据使用指南
===
## 1. 数据清洗
### 1.1 hall_calls_cleaned.csv
The raw hall call dataset contains 259,013 records collected over a one-month period. However, approximately 13.4% of entries (34,724 records) lack valid floor information (Floor is NaN), rendering them unusable for spatial or directional analysis. We adopted the following cleaning protocol:

Remove records with missing Floor: Since the core objective is to model elevator demand by floor and time, any call without a known origin floor cannot contribute to pattern recognition or prediction.
Retain all valid temporal and directional metadata: For remaining records, we preserved Time, Direction ('Up'/'Down'), and derived features (Hour, Weekday, Is_Weekend) without imputation or interpolation, ensuring fidelity to observed behavior.
No outlier removal based on response time: Although some calls have long waiting times (>60s), these are real system behaviors under congestion and were retained to reflect true operational stress.
The resulting cleaned dataset contains 224,289 records, representing 86.6% of the original data. Given that missingness appears random (no systematic bias across hours or weekdays), the retained sample accurately reflects the overall usage patterns of the building. This high retention rate, combined with domain-consistent filtering, ensures both data integrity and representativeness.

我们原始的外呼（Hall Call）数据共 259,013 条，但其中 34,724 条（约 13.4%）缺失“楼层”信息（Floor 为 NaN）。这些记录无法告诉我们“用户从哪一层按了电梯”，因此：

判定为无效的核心原因：建模目标是预测“不同楼层在不同时段的电梯需求”，若不知道起点楼层，则该记录对时空模式挖掘毫无价值。
未采用插值或猜测：我们拒绝用均值、众数或模型填补缺失楼层——因为楼层是离散且结构性强的变量（如1楼和15楼意义完全不同），错误填补会引入严重偏差。
保留所有时间与方向信息：只要楼层存在，无论等待时间多长、发生在几点，我们都保留——因为长等待时间恰恰反映了高峰期的真实拥堵，是宝贵的压力信号。
缺失是否随机？ 我们检查了缺失记录的时间分布，发现其在一天24小时和一周7天中均匀分布，无明显聚集，说明缺失是设备偶发故障或传输丢包所致，非系统性偏差。
最终保留 224,289 条（86.6%），这个比例足够高，且保留的数据完整覆盖了早高峰、午休、晚高峰、工作日/周末等关键场景，能充分代表整栋楼的真实使用行为。因此，清洗后的数据集既干净又不失真，是后续建模的可靠基础。

### 1.3 🧹 Data Cleaning Summary

This document details the data cleaning process for the elevator datasets, ensuring high-quality input for mathematical modeling in traffic prediction, operational mode classification, and dynamic parking strategies. Cleaning was performed using Python with Pandas, focusing on integrity, standardization, and domain-specific validations. Cleaned files are in 'data_cleaning/'. Operations prioritize retaining representative data while removing invalid entries without bias.

本文档详述电梯数据集的数据清洗过程，确保高质量输入用于流量预测、运行模式分类和动态停放策略的数学建模。清洗使用Python的Pandas进行，聚焦完整性、标准化和领域特定验证。清洗文件位于'data_cleaning/'。操作优先保留代表性数据，同时移除无效条目而不引入偏差。

#### Cleaning Principles / 清洗原则
- Integrity: Retain >85% data; remove only invalid (e.g., NaN floors, negative loads). No imputation to avoid artifacts.  
  完整性：保留>85%数据；仅移除无效（如NaN楼层、负负载）。无插值以避免人工偏差。
- Standardization: Convert times to datetime, floors to integers.  
  标准化：时间转为datetime，楼层转为整数。
- Domain Filters: Physical constraints (loads [0,2100] kg); logical inferences (direction from floor changes).  
  领域过滤：物理约束（负载[0,2100] kg）；逻辑推断（方向从楼层变化）。
- Encoding/Sorting: GBK input, UTF-8 output; sort for analysis.  
  编码/排序：GBK输入，UTF-8输出；排序用于分析。

#### File-Specific Operations Table / 各文件操作表格

| File / 文件 | Key Operations / 关键操作 | Techniques & Points (English/Chinese) / 技术要点（英文/中文对照） | Retention Rate & Rationale / 保留率与理由 |
|-------------|---------------------------|----------------------------------------------------------|-------------------------------------------|
| hall_calls.csv | - Standardize time & direction ('Up'/'Down').<br>- Expand multi-floors (e.g., "4,5" to rows).<br>- Safe handling of NaN/invalid floors (retain until final removal). / - 标准化时间与方向（'Up'/'Down'）。<br>- 扩展多楼层（如"4,5"扩展为行）。<br>- 安全处理NaN/无效楼层（保留至最终移除）。 | - Use pd.to_datetime for time parsing; str.capitalize for direction.<br>- Custom function for comma-split expansion, with try-except to avoid data loss.<br>- Points: Handles sensor logging errors (e.g., missing floors from incomplete calls); preserves for spatial analysis. / - 使用pd.to_datetime解析时间；str.capitalize标准化方向。<br>- 自定义函数处理逗号拆分，try-except避免数据丢失。<br>- 要点：处理传感器日志错误（如呼叫不完整导致缺失楼层）；保留用于空间分析。 | 223,339 (86%): High retention ensures accurate demand patterns; removes only true invalids for prediction reliability. / 223,339 (86%)：高保留率确保准确需求模式；仅移除真无效以提升预测可靠性。 |
| car_stops.csv | - Standardize time & floors.<br>- Infer directions from previous floors for garbled entries.<br>- Remove stops with all reasons 'No'. / - 标准化时间与楼层。<br>- 从前一楼层推断乱码方向。<br>- 移除所有原因'No'的停靠。 | - Groupby shift for prev_floor; apply lambda for inference.<br>- Mask filtering for invalid stops.<br>- Points: Addresses 'user left without riding' scenarios; improves mode classification by validating stop reasons. / - Groupby shift获取前楼层；apply lambda推断。<br>- 掩码过滤无效停靠。<br>- 要点：处理“用户按后离开”场景；通过验证停靠原因提升模式分类。 | 214,263 (98%): Minimal loss; inferred directions enhance data usability for trajectory modeling. / 214,263 (98%)：最小丢失；推断方向提升轨迹建模可用性。 |
| load_changes.csv | - Standardize time & floors; rename columns.<br>- Filter loads to [0,2100] kg.<br>- Sort by ID/time. / - 标准化时间与楼层；重命名列。<br>- 过滤负载至[0,2100] kg。<br>- 按ID/时间排序。 | - Boolean masking for range filter.<br>- sort_values for sequencing.<br>- Points: Eliminates sensor noise (negatives/overloads); sorted for time-series flow estimation; based on 2100kg rated capacity. / - 布尔掩码范围过滤。<br>- sort_values排序。<br>- 要点：消除传感器噪声（负值/超载）；排序用于时间序列流量估计；基于2100kg额定容量。 | 216,884 (99%): Preserves valid passenger weights; essential for accurate volume prediction. / 216,884 (99%)：保留有效乘客重量；对准确体积预测至关重要。 |
| maintenance_mode.csv | - Standardize time.<br>- Remove duplicates; sort by ID/time. / - 标准化时间。<br>- 移除重复；按ID/时间排序。 | - drop_duplicates; sort_values.<br>- Points: Ensures paired Enter/Exit (simple state check); excludes maintenance from traffic models to avoid skewed patterns. / - drop_duplicates；sort_values。<br>- 要点：确保Enter/Exit配对（简单状态检查）；从流量模型排除维护以避免偏差模式。 | 161 (100%): Full retention; sorted for easy integration in exclusion filters. / 161 (100%)：全保留；排序便于排除过滤集成。 |
| car_calls.csv | - Standardize time & floors.<br>- Filter actions to 'Register'/'Cancel'. / - 标准化时间与楼层。<br>- 过滤动作至'Register'/'Cancel'。 | - isin for action filter.<br>- Points: Validates internal calls; complements hall_calls for full demand cycles; handles mislogged actions. / - isin动作过滤。<br>- 要点：验证内部呼叫；补充hall_calls完成需求周期；处理误日志动作。 | 255,971 (99%): High quality for response analysis; minimal filtering. / 255,971 (99%)：高质用于响应分析；最小过滤。 |
| car_departures.csv | - Standardize time & floors. / - 标准化时间与楼层。 | - Basic coercion and NaN drop.<br>- Points: Ensures clean trajectories; supports parking strategy by tracking departures. / - 基本强制转换和NaN移除。<br>- 要点：确保干净轨迹；通过跟踪出发支持停放策略。 | 218,491 (100%): No loss; direct usability for optimization models. / 218,491 (100%)：无丢失；直接用于优化模型。 |

#### Notes / 注意事项
- Overall retention >95% average, confirming representative datasets.<br>- Handled issues like sensor faults, user behaviors (e.g., canceled calls), and logging gaps.<br>- For modeling, derive features (e.g., Hour, Weekday) post-cleaning. / - 整体保留率>95%平均，确认代表性数据集。<br>- 处理问题如传感器故障、用户行为（如取消呼叫）和日志间隙。<br>- 对于建模，清洗后派生特征（如Hour, Weekday）。
===


## 2.数据可视化
这里我们共生成了七张图，希望构成“时间 → 方向 → 周期 → 空间 → 负载”, 共同回答：“电梯系统在何时、何地、以何种方向、承受多大压力？”
---
### 2.1 response_time_distribution.pdf 响应时间分布
Establishes baseline system performance; confirms most calls are served quickly (<30s), justifying focus on demand prediction rather than system failure diagnosis.
证明当前系统整体高效（95%响应<30秒），说明问题核心不是“电梯坏了”，而是“如何预判高峰需求以进一步优化调度”。因此，我们的任务应聚焦于需求预测而非故障修复。
---
### 2.2 hourly_calls.pdf 每小时呼叫总量
Reveals bimodal daily demand pattern (peaks at 12:00 and 18:00), indicating strong human activity rhythms driven by lunch breaks and evening departures.
揭示一天中有两个明确高峰（中午12点、傍晚18点），说明电梯使用受人类作息严格支配。这提示我们：时间特征（尤其是小时）是预测的关键输入。
---
### 2.3 hourly_up_and_down 上行、下行呼叫
Identifies classic traffic modes—morning up-peak (7–9 AM) and evening down-peak (5–7 PM)—critical for designing direction-aware dispatching strategies.
展现典型的“早上去楼上、晚上下楼下”模式。这意味着：仅预测“总需求”不够，必须区分方向，否则调度算法会低效（如空梯上行却无人要上）。
---
### 2.4 elevator_usage_by_hours.pdf 平均电梯重量变化量
Validates that call volume peaks correspond to actual loading intensity (e.g., 160 kg at noon), confirming that high call counts reflect genuine passenger load, not spurious button presses.
用“重量”作为真实负载代理变量，证明中午12点不仅是呼叫多，而且确实人多、负载重。排除了“有人乱按按钮”的干扰，确认高峰是真实的。
---
### 2.5 hourly_calls_weekdays_vs_weekend.pdf 工作日v.s.周末呼叫量
Demonstrates stark contrast—structured peaks on weekdays vs flat, low usage on weekends—confirming the building is office-dominated and enabling weekday-specific modeling.
工作日有清晰高峰，周末几乎没人。说明：必须区分工作日/周末，甚至可考虑两套模型。若混在一起训练，会模糊关键模式。
---
### 2.6 & 2.7 hall_calls_heatmap_weekday.pdf & hall_calls_heatmap_weekends.pdf 工作日、周末热力图
Exposes spatial heterogeneity—intense activity on Floors 1–3 (lobby) and 10–14 (office zones) during rush hours on weekdays, versus uniform low usage on weekends. This identifies “hotspot floors” for targeted optimization.
热力图揭示：需求不仅随时间变，也随楼层变。工作日集中在1-3楼（大厅）和10-14楼（办公区），周末则均匀分散。这意味着：预测模型必须包含“楼层”作为特征，且不同楼层的模式不同。
---
### 2.8 summary
These seven visualizations collectively answer the fundamental questions of elevator demand modeling: when, where, in which direction, and how intensely users request service. They avoid redundant or less informative plots (e.g., total calls per floor alone, which ignores time dynamics) and instead emphasize temporal-spatial-directional interactions—the core drivers of elevator traffic. This focused EDA directly informs our feature engineering and model architecture choices.

这种联动，让我们意识到：必须构建一个能同时处理时间（小时、周类型）、空间（楼层）、方向（上/下）的模型。这也解释了为什么我们选择时空序列模型（如LSTM+Embedding）而非简单回归。
===
## 3. 怎么用这几张图？
目前AI是这么建议我们用这几张图的
### 3.1 流量预测
#### 每小时呼叫总量（Hourly Calls）
This plot shows the bimodal daily demand pattern, indicating clear peaks at lunchtime and evening commute. Understanding these patterns is crucial for predicting future traffic volumes.
展示了明显的双峰模式（中午和傍晚高峰），帮助我们识别一天中的关键时段，为时间序列模型提供基础特征。
#### 上行 vs 下行呼叫（Up vs Down by Hour）
Identifies directional traffic modes (morning up-peak and evening down-peak), which are essential for direction-aware prediction models.
显示了方向性需求（早上上行、晚上下行），提示我们需要在预测模型中区分上下行，提升准确性。
#### 工作日 vs 周末呼叫量（Weekday vs Weekend）
Demonstrates stark contrast between weekdays and weekends, highlighting the need for separate models or features to account for day-of-week effects.
工作日和周末的需求差异显著，表明需要考虑周类型作为特征，避免单一模型的偏差。
### 3.2 运行模式分类（建筑脉搏任务）

#### 每小时呼叫总量（Hourly Calls）
Provides a baseline understanding of daily traffic patterns, helping to identify distinct operational modes such as morning peak, lunch rush, and evening departure.
提供了每日交通模式的基础理解，有助于识别不同的运行模式如早高峰、午休、晚高峰等。
#### 上行 vs 下行呼叫（Up vs Down by Hour）
Reveals classic traffic modes like morning up-peak and evening down-peak, critical for classifying building states into predefined modes.
揭示了典型的交通模式（早上上行、晚上下行），是分类当前建筑状态为不同模式的关键依据。
#### 工作日 vs 周末呼叫量（Weekday vs Weekend）
Shows that weekdays exhibit structured demand while weekends have flat, low usage, supporting the classification of operational modes based on weekday/weekend differences.
展现了工作日有规律的需求而周末需求平坦且低，支持基于周类型差异的运行模式分类。
#### 工作日/周末热力图（Spatial-Temporal Heatmaps）
Exposes spatial heterogeneity, identifying hotspot floors during different times of the day, aiding in the identification of specific operational modes.
揭示了空间异质性，识别出特定时间段的热点楼层，有助于发现特定的运行模式。
### 3.3 动态停放策略（战略等待任务）
#### 每小时呼叫总量（Hourly Calls）
Helps determine optimal idle elevator positions by showing when and where demand surges occur.
帮助确定最佳空闲电梯位置，展示何时何地需求激增。
#### 上行 vs 下行呼叫（Up vs Down by Hour）
Informs dynamic parking strategies by highlighting periods with strong directional preferences, guiding decisions on where to park elevators.
通过突出显示强方向偏好期，指导决策在何处停放电梯。
#### 工作日/周末热力图（Spatial-Temporal Heatmaps）
Provides insights into floor-specific demand patterns, informing strategic parking locations and quantities based on historical hotspots.
提供楼层特定需求模式的洞察，根据历史热点指导战略停放位置和数量。
#### 每小时平均进梯重量（Passenger Weight by Hour）
Validates high-demand periods by confirming actual loading intensity, ensuring that dynamic parking strategies target truly busy times.
通过确认实际负载强度验证高需求期，确保动态停放策略针对真正繁忙的时间段。
### 3.4 致管理层备忘录
#### 每小时呼叫总量（Hourly Calls）
Illustrates the necessity of optimizing elevator placement based on observed traffic patterns, providing concrete evidence for management.
通过展示观察到的交通模式，说明优化电梯放置的必要性，为管理层提供具体证据。
#### 工作日 vs 周末呼叫量（Weekday vs Weekend）
Highlights the difference in demand patterns between weekdays and weekends, justifying tailored strategies for each.
强调工作日和周末需求模式的不同，证明为每种情况制定专门策略的合理性。
#### 工作日/周末热力图（Spatial-Temporal Heatmaps）
Visualizes floor-specific demand, making it easier for management to understand why certain floors are prioritized in parking strategies.
可视化楼层特定需求，使管理层更容易理解为何某些楼层在停放策略中被优先考虑。

📊 联合结论（Why These 7? Why Not Others?）
These seven visualizations collectively address the core tasks of traffic prediction, operational mode classification, and dynamic parking strategies. They focus on temporal-spatial-directional interactions—essential factors driving elevator traffic. Other plots, such as total calls per floor alone, would ignore critical time dynamics and fail to provide actionable insights.
这7张图共同解决了流量预测、运行模式分类和动态停放策略的核心任务。它们聚焦于时间-空间-方向的互动——驱动电梯交通的关键因素。其他图表（如仅楼层总呼叫量）会忽略重要的时间动态，无法提供可操作的见解。

在project/situ_model_2026里面的data_cleaning与我们在这里的数据是一模一样的