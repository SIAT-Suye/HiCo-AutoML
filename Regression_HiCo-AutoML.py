from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.feature_selection import RFECV
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt 
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
import time

df = pd.read_csv(r"/kaggle/input//WildBlueberryPollinationSimulationData.csv")
df.drop(['Row', 'fruitset', 'fruitmass', 'seeds'],axis=1,inplace=True)

# 从DataFrame中提取特征名称列表
def load_feature_names_from_df(df):
    """
    从DataFrame中提取特征名称列表。

    参数:
    df (pd.DataFrame): 输入的DataFrame。

    返回:
    list: 特征名称列表。
    """
    feature_names = df.columns.tolist()
    return feature_names

def preprocess_data(X):
    """
    标准化数据。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。

    返回:
    numpy.ndarray: 标准化后的数据矩阵。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def find_optimal_clusters(X):
    """
    确定最优的聚类数量。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。

    返回:
    int: 最优的聚类数量。
    """
    max_clusters = X.shape[1]
    best_score = -1
    optimal_clusters = 2

    for n_clusters in range(2, max_clusters):
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(X.T)
        score = silhouette_score(X.T, cluster_labels) - davies_bouldin_score(X.T, cluster_labels)
        if score > best_score:
            best_score = score
            optimal_clusters = n_clusters

    return optimal_clusters

def feature_clustering(X, n_clusters):
    """
    对特征进行聚类。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    n_clusters (int): 聚类数量。

    返回:
    numpy.ndarray: 每个特征的聚类标签。
    """
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clustering.fit_predict(X.T)
    
    # 打印聚类信息
    print(f"Cluster number: {n_clusters}")
    for cluster_id in range(n_clusters):
        cluster_features = np.where(cluster_labels == cluster_id)[0]
        print(f"Cluster {cluster_id}: Feature index {cluster_features}")
    
    return cluster_labels

def select_features_within_clusters(X, y, cluster_labels):
    """
    在每个聚类中选择特征。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    y (numpy.ndarray): 目标标签。
    cluster_labels (numpy.ndarray): 每个特征的聚类标签。

    返回:
    dict: 每个聚类中选中的特征索引。
    """
    models = {
        'LGBM': LGBMRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(random_state=42),
        'GradientBoost': GradientBoostingRegressor(random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGB': XGBRegressor(random_state=42)
    }

    selected_features = {}

    for cluster_id in set(cluster_labels):
        cluster_features = np.where(cluster_labels == cluster_id)[0]
        X_cluster = X[:, cluster_features]
        selected_features[cluster_id] = {}

        if len(cluster_features) > 1:
            for model_name, model in models.items():
                selector = RFECV(model, cv=5)
                selector.fit(X_cluster, y)
                selected_indices = selector.support_
                selected_features[cluster_id][model_name] = cluster_features[selected_indices]
                score = np.mean(cross_val_score(model, X_cluster[:, selected_indices], y, cv=5, scoring='r2'))
                print(f"Cluster {cluster_id}, Model {model_name}: Selected Features Within Clusters {cluster_features[selected_indices]}, Score {score:.4f}")
        else:
            for model_name in models:
                selected_features[cluster_id][model_name] = cluster_features

    return selected_features

def select_features_across_clusters(X, y, selected_features):
    """
    在所有选中的特征中进行最终选择。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    y (numpy.ndarray): 目标标签。
    selected_features (dict): 每个聚类中选中的特征索引。

    返回:
    tuple: 最优模型和对应的特征子集。
    """
    models = {
        'LGBM': LGBMRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(random_state=42),
        'GradientBoost': GradientBoostingRegressor(random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGB': XGBRegressor(random_state=42)
    }

    best_model = None
    best_score = -1
    best_features = []
    best_feature_count = float('inf')  # 初始化为无穷大

    for model_name, model in models.items():
        all_selected_features = []
        for cluster_id in selected_features:
            if model_name in selected_features[cluster_id]:
                all_selected_features.extend(selected_features[cluster_id][model_name])

        if not all_selected_features:
            continue

        X_selected = X[:, all_selected_features]
        selector = RFECV(model, cv=5)
        selector.fit(X_selected, y)
        selected_indices = selector.support_
        current_features = [all_selected_features[i] for i in np.where(selected_indices)[0]]
        score = np.mean(cross_val_score(model, X[:, current_features], y, cv=5, scoring='r2'))
        print(f"Model {model_name}: Selected Features {current_features}, Score {score:.4f}")

        if score > best_score or (score == best_score and len(current_features) < best_feature_count):
            best_score = score
            best_model = model
            best_features = current_features
            best_feature_count = len(current_features)

    # 训练最佳模型并提取特征重要性得分
    best_model.fit(X[:, best_features], y)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importances = best_model.coef_
    else:
        feature_importances = None
        
    # 加载特征名称列表
    feature_names = load_feature_names_from_df(df)

    if feature_importances is not None:
        print("Feature importance value of best model:")
        for feature_index, importance in zip(best_features, feature_importances):
            feature_name = feature_names[feature_index]
            print(f"Feature {feature_name}: Importance value {importance:.4f}")

    return best_model, best_features

def bayesian_optimization(model, X, y):
    if isinstance(model, SVC):
        def svm_objective(C, gamma):
            model.set_params(C=C, gamma=gamma)
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            return np.mean(scores)

        optimizer = BayesianOptimization(
            f=svm_objective,
            pbounds={'C': (0.1, 100), 'gamma': (0.001, 1)},
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=25)
        best_params = optimizer.max['params']
        best_params['C'] = best_params['C']
        best_params['gamma'] = best_params['gamma']
        
    elif isinstance(model, (XGBClassifier, LGBMClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, DecisionTreeClassifier, XGBRegressor, LGBMRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, DecisionTreeRegressor)):
        def tree_objective(n_estimators, max_depth):
            model.set_params(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth)
            )
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            return np.mean(scores)

        optimizer = BayesianOptimization(
            f=tree_objective,
            pbounds={
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
            },
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=25)
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        
    elif isinstance(model, LogisticRegression):
        def lr_objective(C):
            model.set_params(C=C)
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            return np.mean(scores)

        optimizer = BayesianOptimization(
            f=lr_objective,
            pbounds={'C': (0.1, 100)},
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=25)
        best_params = optimizer.max['params']
        best_params['C'] = best_params['C']
    else:
        raise ValueError("不支持的模型类型")

    return best_params

def HiCo_AutoML(X, y):
    """
    HiCo-AutoML算法。

    参数:
    X (numpy.ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    y (numpy.ndarray): 目标标签。

    返回:
    tuple: 最优模型和对应的特征子集以及模型参数。
    """
    X_scaled = preprocess_data(X)
    n_clusters = find_optimal_clusters(X_scaled)
    cluster_labels = feature_clustering(X_scaled, n_clusters)
    selected_features = select_features_within_clusters(X_scaled, y, cluster_labels)
    best_model, best_features = select_features_across_clusters(X_scaled, y, selected_features)
    best_params = bayesian_optimization(best_model, X_scaled[:, best_features], y)
    
    return best_model, best_features, best_params

X = df.drop(['Yield'], axis = 1).values
y = df['Yield'].values

# 定义实验次数
num_experiments = 10
elapsed_times = []  # 用于存储每次实验的运行时间
for _ in range(num_experiments):
    # 记录开始时间
    start_time = time.time()
    best_model, best_features, best_params = HiCo_AutoML(X, y)
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)
    print(f"代码运行时间: {elapsed_time} 秒")
# 计算10次实验的均值
average_time = np.mean(elapsed_times)
print(f"代码运行时间: {average_time} 秒")
print("Best_model:", best_model)
print("Best_features:", best_features)
print("Best_params:", best_params)

# 使用选择的特征进行训练和预测
feature_fs = df.columns.drop(['Yield'])[[best_features]].values
feature_fs  # 获取选择的特征子集
fs_df = pd.concat([df[feature_fs], df['Yield']], axis=1)

# 加载数据集并进行标准化处理
X = fs_df.drop(['Yield'], axis = 1).values
y = fs_df['Yield'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_model = GradientBoostingRegressor(max_depth=3, n_estimators=296, random_state=42)  
best_model.fit(X_train, y_train)  
y_pred = best_model.predict(X_test)  

y_mean = np.mean(y_test)  

  
# 定义评估函数  
def evaluate_model(y_true, y_pred):  
    r2 = r2_score(y_true, y_pred)  
    mae = mean_absolute_error(y_true, y_pred)  
    mse = mean_squared_error(y_true, y_pred)  
    rmse = np.sqrt(mse)  
    rrmse = rmse / (y_mean if y_mean != 0 else 1)   
    return r2, mae, mse, rmse, rrmse  

r2, mae, mse, rmse, rrmse = evaluate_model(y_test, y_pred)  
print(f"HiCo-AutoML: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, RRMSE: {rrmse:.4f}")

