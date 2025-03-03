!pip3 install auto-sklearn
! pip install tpot
! pip install h2o
! pip install autogluon
! pip install autokeras

try :
    import autosklearn
except :
    try :
        import autosklearn
    except :
        pass

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.feature_selection import RFECV
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from lightgbm import LGBMClassifier,  LGBMRegressor
from bayes_opt import BayesianOptimization

# Blueberry Yield
df = pd.read_csv(r"/kaggle/input/blueberry/WildBlueberryPollinationSimulationData.csv")
df.drop(['Row', 'fruitset', 'fruitmass', 'seeds'],axis=1,inplace=True)

X = df.drop(['Yield'], axis = 1).values
y = df['Yield'].values

# PM2.5 Concentration
# df = pd.read_csv(r"/kaggle/input/pm2-5-prediction/shenzhen_pm2.5.csv", names=['date','quality','AQI','ranking','PM2.5(μg/m3)','PM10(μg/m3)','SO2(μg/m3)','NO2(μg/m3)','CO(mg/m3)','O3(μg/m3)'])
# df.drop(['date','quality'],axis=1,inplace=True)

# X = df.drop(['PM2.5(μg/m3)'], axis = 1).values
# y = df['PM2.5(μg/m3)'].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):  
    r2 = r2_score(y_true, y_pred)  
    mae = mean_absolute_error(y_true, y_pred)  
    mse = mean_squared_error(y_true, y_pred)  
    rmse = np.sqrt(mse)  
    return r2, mae, mse, rmse 

import numpy as np
import autosklearn.regression
from sklearn.metrics import r2_score, mean_squared_error

# 定义实验次数
num_experiments = 10
elapsed_times = []  
r2_scores = []
rmse_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 使用 AutoSklearn 进行自动机器学习，增加内存限制参数
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=60 * 2,  
        per_run_time_limit=10,          
        metric=autosklearn.metrics.r2,  
        seed=42
    )
    automl.fit(X_train, y_train)

    # 输出模型信息
    automl.leaderboard(detailed=True, ensemble_only=False)
    print(automl.sprint_statistics())
    print(automl.show_models())

    # 进行预测并评估
    predictions = automl.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # 存储每次实验的指标
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间并添加到列表
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
    elapsed_times.append(elapsed_time)

# 计算实验的均值
average_time = np.mean(elapsed_times)
average_r2 = np.mean(r2_scores)
average_rmse = np.mean(rmse_scores)

# 输出均值
print(f"AutoSklearn 平均运行时间: {average_time:.4f} 秒")
print(f"AutoSklearn 平均 R² 值: {average_r2:.4f}, 平均 RMSE: {average_rmse:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体样式  
plt.rcParams['font.family'] = 'Times New Roman'   
  
# 使用 jointplot  
g = sns.jointplot(x=y_test, y=predictions, kind="reg", color='green', height=8)  # 可以添加一个 height 参数来调整图形大小  
  
# 设置 x 轴和 y 轴的标签  
g.set_axis_labels('Actual PM2.5', 'Predicted PM2.5', fontsize=20)  # 使用 set_axis_labels 设置字体大小  

# 设置 x 轴的刻度  
xticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_xticks(xticks)  # 设置散点图 x 轴的刻度  
g.ax_marg_x.set_xticks(xticks)  # 设置 x 轴边际分布的刻度  

yticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_yticks(yticks)   
g.ax_marg_y.set_yticks(yticks)
  
# 设置刻度标签大小（需要分别设置 x 和 y 轴的刻度）  
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:  
    ax.tick_params(labelsize=20)  # 设置所有轴的刻度标签大小  
    
# 关闭网格  
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
    
# 添加绿色的虚线对角线  
# 假设 x 和 y 的范围大致相同，我们可以使用 min 和 max 来确定对角线的范围  
xmin, xmax = g.ax_joint.get_xlim()  
ymin, ymax = g.ax_joint.get_ylim()  
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'b--', linewidth=2)    
    
plt.show()

import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score, mean_squared_error
from autogluon.tabular import TabularPredictor

# 定义评估模型的函数
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse

# 记录开始时间
start_time = time.time()

# 将训练集和测试集的特征转换为 DataFrame
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# 如果 y_train 和 y_test 是一维数组，可以将它们转换为 Series
y_train_df = pd.Series(y_train, name="Yield")
y_test_df = pd.Series(y_test, name="Yield")

# 合并特征和目标以供 AutoGluon 使用
train_data = pd.concat([X_train_df.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)

# 使用 AutoGluon 进行建模
predictor = TabularPredictor(label='Yield', eval_metric='r2').fit(
    train_data,
    time_limit=120,  
    auto_stack=True,
    excluded_model_types=['NN_TORCH'],
    fit_weighted_ensemble=False  
)

# 进行预测
y_pred = predictor.predict(test_data.drop(['Yield'], axis=1))
y_true = test_data['Yield'].values

# 记录结束时间
end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f"代码运行时间: {elapsed_time} 秒")

# 计算评估指标
r2, rmse = evaluate_model(y_true, y_pred)
print(f"AutoGluon R²: {r2:.4f}, RMSE: {rmse:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体样式  
plt.rcParams['font.family'] = 'Times New Roman'   
  
# 使用 jointplot  
g = sns.jointplot(x=y_true, y=y_pred, kind="reg", color='green', height=8)  # 可以添加一个 height 参数来调整图形大小  
  
# 设置 x 轴和 y 轴的标签  
g.set_axis_labels('Actual PM2.5', 'Predicted PM2.5', fontsize=20)  # 使用 set_axis_labels 设置字体大小  

# 设置 x 轴的刻度  
xticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_xticks(xticks)  # 设置散点图 x 轴的刻度  
g.ax_marg_x.set_xticks(xticks)  # 设置 x 轴边际分布的刻度  

yticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_yticks(yticks)   
g.ax_marg_y.set_yticks(yticks)
  
# 设置刻度标签大小（需要分别设置 x 和 y 轴的刻度）  
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:  
    ax.tick_params(labelsize=20)  # 设置所有轴的刻度标签大小  
    
# 关闭网格  
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
    
# 添加绿色的虚线对角线  
# 假设 x 和 y 的范围大致相同，我们可以使用 min 和 max 来确定对角线的范围  
xmin, xmax = g.ax_joint.get_xlim()  
ymin, ymax = g.ax_joint.get_ylim()  
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'b--', linewidth=2)    
    
plt.show()

import time
import numpy as np
from tpot import TPOTRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 定义实验次数
num_experiments = 10
elapsed_times = []
r2_scores = []
rmse_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 使用 TPOT 进行自动机器学习（回归器版本）
    tpot = TPOTRegressor(generations=5, population_size=8, random_state=42, verbosity=0)
    tpot.fit(X_train, y_train)

    # 进行预测
    y_pred_tpot = tpot.predict(X_test)

    # 评估模型
    r2 = r2_score(y_test, y_pred_tpot)
    rmse = mean_squared_error(y_test, y_pred_tpot, squared=False)

    # 存储每次实验的指标
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间并添加到列表
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
    elapsed_times.append(elapsed_time)

# 计算实验的均值
average_time = np.mean(elapsed_times)
average_r2 = np.mean(r2_scores)
average_rmse = np.mean(rmse_scores)

# 输出均值
print(f"TPOT 平均运行时间: {average_time:.4f} 秒")
print(f"TPOT 平均 R² 值: {average_r2:.4f}, 平均 RMSE: {average_rmse:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体样式  
plt.rcParams['font.family'] = 'Times New Roman'   
  
# 使用 jointplot  
g = sns.jointplot(x=y_test, y=y_pred_tpot, kind="reg", color='green', height=8)  # 可以添加一个 height 参数来调整图形大小  
  
# 设置 x 轴和 y 轴的标签  
g.set_axis_labels('Actual PM2.5', 'Predicted PM2.5', fontsize=20)  # 使用 set_axis_labels 设置字体大小  

# 设置 x 轴的刻度  
xticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_xticks(xticks)  # 设置散点图 x 轴的刻度  
g.ax_marg_x.set_xticks(xticks)  # 设置 x 轴边际分布的刻度  

yticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_yticks(yticks)   
g.ax_marg_y.set_yticks(yticks)
  
# 设置刻度标签大小（需要分别设置 x 和 y 轴的刻度）  
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:  
    ax.tick_params(labelsize=20)  # 设置所有轴的刻度标签大小  
    
# 关闭网格  
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
    
# 添加绿色的虚线对角线  
# 假设 x 和 y 的范围大致相同，我们可以使用 min 和 max 来确定对角线的范围  
xmin, xmax = g.ax_joint.get_xlim()  
ymin, ymax = g.ax_joint.get_ylim()  
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'b--', linewidth=2)    
    
plt.show()

import h2o
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 定义实验次数
num_experiments = 10
elapsed_times = []
r2_scores = []
rmse_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 初始化 H2O
    h2o.init()

    # 创建 H2O 的数据集，并命名目标列为 'target'
    h2o_df = h2o.H2OFrame(np.hstack((X_train, y_train.reshape(-1, 1))),
                          column_names=[f'feature_{i}' for i in range(X_train.shape[1])] + ['Yield'])
    h2o_df_test = h2o.H2OFrame(np.hstack((X_test, y_test.reshape(-1, 1))),
                               column_names=[f'feature_{i}' for i in range(X_test.shape[1])] + ['Yield'])

    # 训练 H2O AutoML 模型（回归任务）
    aml = h2o.automl.H2OAutoML(max_runtime_secs=120, seed=42, project_name='regression_project')
    aml.train(x=h2o_df.columns[:-1], y='Yield', training_frame=h2o_df)

    # 进行预测
    preds = aml.predict(h2o_df_test[:, :-1])
    y_pred_h2o = preds.as_data_frame().values.flatten()  # 使用预测值列

    # 评估模型
    r2 = r2_score(y_test, y_pred_h2o)
    rmse = mean_squared_error(y_test, y_pred_h2o, squared=False)

    # 存储每次实验的指标
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.4f} 秒")
    elapsed_times.append(elapsed_time)

# 计算实验的均值
average_time = np.mean(elapsed_times)
average_r2 = np.mean(r2_scores)
average_rmse = np.mean(rmse_scores)

# 关闭 H2O 实例
h2o.shutdown(prompt=False)

# 输出均值
print(f"平均运行时间: {average_time:.4f} 秒")
print(f"H2O AutoML 平均 R² 值: {average_r2:.4f}, 平均 RMSE: {average_rmse:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
  
# 使用 jointplot  
g = sns.jointplot(x=y_test, y=y_pred_h2o, kind="reg", color='green', height=8)  # 可以添加一个 height 参数来调整图形大小  
  
# 设置 x 轴和 y 轴的标签  
g.set_axis_labels('Actual PM2.5', 'Predicted PM2.5', fontsize=20)  # 使用 set_axis_labels 设置字体大小  

# 设置 x 轴的刻度  
xticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_xticks(xticks)  # 设置散点图 x 轴的刻度  
g.ax_marg_x.set_xticks(xticks)  # 设置 x 轴边际分布的刻度  

yticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_yticks(yticks)   
g.ax_marg_y.set_yticks(yticks)
  
# 设置刻度标签大小（需要分别设置 x 和 y 轴的刻度）  
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:  
    ax.tick_params(labelsize=20)  # 设置所有轴的刻度标签大小  
    
# 关闭网格  
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
    
# 添加绿色的虚线对角线  
# 假设 x 和 y 的范围大致相同，我们可以使用 min 和 max 来确定对角线的范围  
xmin, xmax = g.ax_joint.get_xlim()  
ymin, ymax = g.ax_joint.get_ylim()  
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'b--', linewidth=2)    
    
plt.show()

import autokeras as ak
import time
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

# 转换数据为float32格式
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# 定义评估模型的函数
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse

# 记录开始时间
start_time = time.time()

# 使用 AutoKeras 定义回归模型
regressor = ak.StructuredDataRegressor(
    max_trials=20,          # 最大搜索次数
    overwrite=True,         # 覆盖之前的训练模型
    loss="mean_squared_error"  # 使用均方误差损失函数
)

# 训练模型
regressor.fit(X_train, y_train, epochs=20)

# 测试模型
y_pred = regressor.predict(X_test).ravel()  # 使用 .ravel() 将预测结果转为一维

# 模型评估
r2, rmse = evaluate_model(y_test, y_pred)
print(f"Auto-Keras R²: {r2:.4f}, RMSE: {rmse:.4f}")

# 记录结束时间
end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f"代码运行时间: {elapsed_time} 秒")

import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体样式  
plt.rcParams['font.family'] = 'Times New Roman'   
  
# 使用 jointplot  
g = sns.jointplot(x=y_test, y=y_pred, kind="reg", color='green', height=8)  # 可以添加一个 height 参数来调整图形大小  
  
# 设置 x 轴和 y 轴的标签  
g.set_axis_labels('Actual PM2.5', 'Predicted PM2.5', fontsize=20)  # 使用 set_axis_labels 设置字体大小  

# 设置 x 轴的刻度  
xticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_xticks(xticks)  # 设置散点图 x 轴的刻度  
g.ax_marg_x.set_xticks(xticks)  # 设置 x 轴边际分布的刻度  

yticks = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]  
g.ax_joint.set_yticks(yticks)   
g.ax_marg_y.set_yticks(yticks)
  
# 设置刻度标签大小（需要分别设置 x 和 y 轴的刻度）  
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:  
    ax.tick_params(labelsize=20)  # 设置所有轴的刻度标签大小  
    
# 关闭网格  
g.ax_joint.grid(False)
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
    
# 添加绿色的虚线对角线  
# 假设 x 和 y 的范围大致相同，我们可以使用 min 和 max 来确定对角线的范围  
xmin, xmax = g.ax_joint.get_xlim()  
ymin, ymax = g.ax_joint.get_ylim()  
g.ax_joint.plot([xmin, xmax], [ymin, ymax], 'b--', linewidth=2)    
    
plt.show()
