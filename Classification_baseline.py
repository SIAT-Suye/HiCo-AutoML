df_fire = pd.read_csv(r"/kaggle/input/forest-fire-risk-prediction/gds_fire_30mm.csv")
df_nofire = pd.read_csv(r"/kaggle/input/forest-fire-risk-prediction/gds_nofire_30mm.csv")
df = pd.concat([df_fire,df_nofire])

df = df.rename(columns={'system:index': 'System:index', 'aspect': 'Aspect', 'days': 'Days', 'elevation': 'Elevation', 'evimax': 'EVImax', 
                        'evimean': 'EVImean','evimedian': 'EVImedian', 'evimin': 'EVImin', 'evisum': 'EVIsum', 'labels': 'Labels', 'lstmax': 'LSTmax', 'lstmean': 'LSTmean',
                        'lstmedian': 'LSTmedian', 'lstmin': 'LSTmin', 'lstsum': 'LSTsum', 'ndvimax': 'NDVImax', 'ndvimean': 'NDVImean', 'ndvimedian': 'NDVImedian',
                        'ndvimin': 'NDVImin', 'ndvisum': 'NDVIsum', 'ndwimax': 'NDWImax', 'ndwimean': 'NDWImean', 'ndwimedian': 'NDWImedian', 'ndwimin': 'NDWImin',
                        'ndwisum': 'NDWIsum', 'precmax': 'PRECmax', 'precmean': 'PRECmean', 'precmedian': 'PRECmedian', 'precmin': 'PRECmin', 'precsum': 'PRECsum', 'slope': 'Slope', 'time':  'Time'})

features = ['System:index', 'Time', 'Days', 'EVIsum', 'NDWIsum', 'NDVIsum', 'LSTsum', '.geo']

df = df.drop(features, axis = 1)
df = df[df.apply(lambda row: row.isin([-99999.000000]).sum() == 0, axis=1)]

# Water Quality
# df = pd.read_csv(r"/kaggle/input//waterQuality.csv")
# df = df.drop(df[df.is_safe=='#NUM!'].index) # 丢弃3项缺失值
# df['is_safe'] = df['is_safe'].astype('int64')
# df['ammonia'] = df['ammonia'].astype('float64')

# Diabetes Diagnosis 
# df = pd.read_csv(r"/kaggle/input//diabetes_data.csv")
# df.drop(['PatientID','DoctorInCharge'],axis=1,inplace=True)

X = df.drop(['Labels'], axis = 1).values
y = df['Labels'].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义评估函数  
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

import time
import numpy as np
import autosklearn.classification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# 定义实验次数
num_experiments = 10
elapsed_times = []  
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 使用 AutoSklearn 进行自动机器学习，增加内存限制参数
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60 * 10,  
        per_run_time_limit=30,          
        metric=autosklearn.metrics.accuracy,  
        seed=42,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )
    automl.fit(X_train, y_train)
    # 输出模型信息
    automl.leaderboard(detailed=True, ensemble_only=False)
    print(automl.sprint_statistics())
    print(automl.show_models())

    # 获取预测结果和概率
    predictions = automl.predict(X_test)
    probabilities = automl.predict_proba(X_test)[:, 1]  # 获取类别1的预测概率

    # 评估模型性能
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    # 存储每次实验的指标
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间并添加到列表
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
    elapsed_times.append(elapsed_time)

# 计算平均指标
average_time = np.mean(elapsed_times)
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_roc_auc = np.mean(roc_auc_scores)

# 输出平均指标
print(f"AutoSklearn 平均运行时间: {average_time:.4f} 秒")
print(f"AutoSklearn 平均准确率: {average_accuracy:.4f}, 平均精确率: {average_precision:.4f}, 平均召回率: {average_recall:.4f}, 平均 F1 值: {average_f1:.4f}, 平均 ROC-AUC: {average_roc_auc:.4f}")

import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

# 定义评估模型的函数
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

# 进行 10 次实验
num_experiments = 10
accuracy_list, precision_list, recall_list, f1_list, roc_auc_list, time_list = [], [], [], [], [], []

for i in range(num_experiments):
    print(f"\n正在运行实验 {i+1}/{num_experiments}...")
    start_time = time.time()
    
    # 将训练集和测试集的特征转换为 DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # 如果 y_train 和 y_test 是一维数组，可以将它们转换为 Series
    y_train_df = pd.Series(y_train, name="Labels")
    y_test_df = pd.Series(y_test, name="Labels")
    
    # 合并特征和目标以供 AutoGluon 使用
    train_data = pd.concat([X_train_df.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
    
    # 使用 AutoGluon 进行建模
    predictor = TabularPredictor(label='Labels', eval_metric='accuracy').fit(
        train_data,
        time_limit=600,  
        auto_stack=True,
        excluded_model_types=['NN_TORCH'],
        fit_weighted_ensemble=False  
    )
    
    # 进行预测
    y_pred = predictor.predict(test_data.drop(['Labels'], axis=1))
    y_prob = predictor.predict_proba(test_data.drop(['Labels'], axis=1))[1]  # 获取类别1的预测概率
    y_true = test_data['Labels'].values
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    time_list.append(elapsed_time)
    
    # 计算评估指标
    accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # 记录结果
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)
    
    print(f"实验 {i+1} 完成: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Time={elapsed_time:.2f} 秒")

# 计算均值
accuracy_mean = np.mean(accuracy_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
f1_mean = np.mean(f1_list)
roc_auc_mean = np.mean(roc_auc_list)
time_mean = np.mean(time_list)

# 输出最终均值结果
print("\n===== 10 次实验的平均结果 =====")
print(f"平均 Accuracy: {accuracy_mean:.4f}")
print(f"平均 Precision: {precision_mean:.4f}")
print(f"平均 Recall: {recall_mean:.4f}")
print(f"平均 F1 Score: {f1_mean:.4f}")
print(f"平均 ROC-AUC: {roc_auc_mean:.4f}")
print(f"平均运行时间: {time_mean:.2f} 秒")


import time
import numpy as np
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 定义实验次数
num_experiments = 10
elapsed_times = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 使用 TPOT 进行自动机器学习（分类器版本）
    tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=0)
    tpot.fit(X_train, y_train)

    # 进行预测和概率计算
    y_pred_tpot = tpot.predict(X_test)
    y_prob_tpot = tpot.predict_proba(X_test)[:, 1]  # 获取类别1的预测概率

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_tpot)
    precision = precision_score(y_test, y_pred_tpot)
    recall = recall_score(y_test, y_pred_tpot)
    f1 = f1_score(y_test, y_pred_tpot)
    roc_auc = roc_auc_score(y_test, y_prob_tpot)

    # 存储每次实验的指标
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间并添加到列表
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")
    elapsed_times.append(elapsed_time)

# 计算平均指标
average_time = np.mean(elapsed_times)
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_roc_auc = np.mean(roc_auc_scores)

# 输出平均指标
print(f"TPOT 平均运行时间: {average_time:.4f} 秒")
print(f"TPOT 平均准确率: {average_accuracy:.4f}, 平均精确率: {average_precision:.4f}, 平均召回率: {average_recall:.4f}, 平均 F1 值: {average_f1:.4f}, 平均 ROC-AUC: {average_roc_auc:.4f}")

import h2o
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 定义实验次数
num_experiments = 10
elapsed_times = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()

    # 初始化 H2O
    h2o.init()

    # 创建 H2O 的数据集，并手动命名目标列为 'Labels'
    h2o_df = h2o.H2OFrame(np.hstack((X_train, y_train.reshape(-1, 1))),
                          column_names=[f'feature_{i}' for i in range(X_train.shape[1])] + ['Labels'])
    h2o_df_test = h2o.H2OFrame(np.hstack((X_test, y_test.reshape(-1, 1))),
                               column_names=[f'feature_{i}' for i in range(X_test.shape[1])] + ['Labels'])

    # 将目标列 'Labels' 设置为分类
    h2o_df['Labels'] = h2o_df['Labels'].asfactor()
    h2o_df_test['Labels'] = h2o_df_test['Labels'].asfactor()

    # 训练 H2O AutoML 模型
    aml = h2o.automl.H2OAutoML(max_runtime_secs=600, seed=42, project_name='classification_project')
    aml.train(x=h2o_df.columns[:-1], y='Labels', training_frame=h2o_df)

    # 进行预测（包括预测类别和概率）
    preds = aml.predict(h2o_df_test[:, :-1])
    y_pred_h2o = preds['predict'].as_data_frame().values.flatten().astype(int)  # 使用预测的类别
    y_prob_h2o = preds['p1'].as_data_frame().values.flatten()  # 类别1的预测概率

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_h2o)
    precision = precision_score(y_test, y_pred_h2o, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred_h2o, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred_h2o, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob_h2o)

    # 存储每次实验的指标
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.4f} 秒")
    elapsed_times.append(elapsed_time)

# 计算平均指标
average_time = np.mean(elapsed_times)
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_roc_auc = np.mean(roc_auc_scores)

# 关闭 H2O 实例
h2o.shutdown(prompt=False)

# 输出平均指标
print(f"平均运行时间: {average_time:.4f} 秒")
print(f"H2O AutoML 平均准确率: {average_accuracy:.4f}, 平均精确率: {average_precision:.4f}, 平均召回率: {average_recall:.4f}, 平均 F1 值: {average_f1:.4f}, 平均 ROC-AUC: {average_roc_auc:.4f}")

import autokeras as ak
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 定义评估模型的函数
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

# 进行 10 次实验
num_experiments = 10
accuracy_list, precision_list, recall_list, f1_list, roc_auc_list, time_list = [], [], [], [], [], []

for i in range(num_experiments):
    print(f"\n正在运行实验 {i+1}/{num_experiments}...")
    start_time = time.time()
    
    # 使用AutoKeras定义分类模型
    clf = ak.StructuredDataClassifier(
        max_trials=20,          # 最大搜索次数
        overwrite=True,         # 覆盖之前的训练模型
        loss="binary_crossentropy",  # 使用二分类交叉熵损失函数
        metrics=["accuracy"]
    )
    
    # 训练模型
    clf.fit(X_train, y_train, epochs=50, verbose=0)
    
    # 预测
    y_pred = clf.predict(X_test).flatten()
    y_prob = clf.predict(X_test, output_probabilities=True).flatten()
    
    # 计算时间
    elapsed_time = time.time() - start_time
    time_list.append(elapsed_time)
    
    # 评估模型
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # 记录结果
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)
    
    print(f"实验 {i+1} 完成: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Time={elapsed_time:.2f} 秒")

# 计算均值
accuracy_mean = np.mean(accuracy_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
f1_mean = np.mean(f1_list)
roc_auc_mean = np.mean(roc_auc_list)
time_mean = np.mean(time_list)

# 输出最终均值结果
print("\n===== 10 次实验的平均结果 =====")
print(f"平均 Accuracy: {accuracy_mean:.4f}")
print(f"平均 Precision: {precision_mean:.4f}")
print(f"平均 Recall: {recall_mean:.4f}")
print(f"平均 F1 Score: {f1_mean:.4f}")
print(f"平均 ROC-AUC: {roc_auc_mean:.4f}")
print(f"平均运行时间: {time_mean:.2f} 秒")
