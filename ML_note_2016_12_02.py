#机器学习笔记

************  数据预处理  ************

1.特征规范化
把每个特征的值域规范化为0到1之间
from sklearn.preprocessing import MinMaxScaler
X_transformed = MinMaxScaler().fit_transform(X)

使每条数据各特征值的和为1： sklearn.preprocessing.Normalizer
使各特征的均值为0，方差为1: sklearn.preprocessing.StandardScaler (常用作规范化的基准)
数值型特征的二值化： sklearn.preprocessing.Binarizer (大于阈值为1，小于阈值为0)

2.流水线（pipeline）
流水线将对数据的操作依次串联起来，做各种必要的预处理，流水线的输入是一连串的数据挖掘步骤，其中最
后一步必须是估计器。输入的数据集经过转换器处理后，输出的结果可以作为下一步的输入。最后，用位于流
水线最后一步的估计器对数据进行分类。
from sklearn.pipeline import Pipeline
scaling_pipeline = Pipeline([('scale',MinMaxScaler()),
							 ('predict',KNeighborsClassifier())])
scores = cross_val_score(scaling_pipeline, X, y, scoring='accuracy')

3.数据类型转换
	(1)LabelEncoder 将字符串类型转换成整型 #假设有️17支球队，将球队名字（字符串类型）转换成整型0～16
	from sklearn.preprocessing import LabelEncoder
	encoder = LabelEncoder()
	encoder.fit(dataset["HomeTeam"].values)
	home_teams = encoder.transform(dataset["HomeTeam"].values)
	visitor_teams = encoder.transform(dataset["HomeTeam"].values)
	X_teams = np.vstack([home_teams,visitor_teams]).T

	(2)OneHotEncoder 将整数转换成二进制数字 #17支球队已经转换成0～16了，对数值为7的球队，OneHotEncoder为其分配的二进制数
	#的第七位为1其他位为0，其他球队对应的二进制数的第七位为0，所以会形成稀疏矩阵，要用todense()函数转成非稀疏矩阵
	from sklearn.preprocessing import OneHotEncoder
	onehot = OneHotEncoder()
	X_teams = onehot.fit_transform(X_teams).todense()






************  数据集切分  ************

1.随机切分
from sklearn.cross_validation import train_test_split
默认是将数据集的25%作为测试集。
原数据集为X, y:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 14)
将 random_state 的值设置为 None, 每次切分的结果会真正随机。

2.交叉检验
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator, X, y, scoring = 'accuracy')
avg_score = np.mean(scores)*100
print "the average accuracy is {0:.1f}%".format(avg_score)







************  分类器  ************

sklearn的分类器主要有两个函数： fit() 和 predict()。

1.K近邻
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()  #参数默认，选择5个近邻
#选择参数： 
#estimator = KNeighborsClassifier(n_beighbors=n_neighbors)
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predict)*100
print "the accuracy is {0:.1f}%".format(accuracy)

2.决策树
scikit-learn库实现了分类回归树（CART），并将其作为生成决策树的默认算法，支持连续型和类别型特征
决策树的参数：
	（1）退出方法（包含剪枝，即先构造树，再进行修剪，去掉没有提供太多信息的节点）
		min_samples_split: 指定创建一个新节点至少需要的个体数量
		min_samples_leaf : 指定为了保留节点，每个节点至少应包含的个体数量
		#第一个参数控制决策节点的创建，第二个参数决定决策节点是否能保留
	（2）创建决策树的标准
		基尼不纯度：用来衡量决策节点错误预测新个体类别的比例
		信息增益 ：用信息论中的熵来表示决策节点提供多少新信息

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = 14)
scores = cross_val_score(clf,X,y,scoring = "accuracy")
print "the average accuracy is {0:.1f}%".format(np.mean(scores)*100)

3.网格搜索
from sklearn.grid_search import GridSearchCV
parameter_space = {
                   "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   }
clf = DecisionTreeClassifier(random_state = 14)
grid = GridSearchCV(clf, parameter_space)
gird.fit(X,y)
print "Accuracy:{0:.1f}% ".format(grid.best_score_ * 100)

4.随机森林
决策树可以学习很复杂的规则，但容易导致过拟合，解决方法之一是调整决策树算法，比如限制决策树的深度来限制它学到规则的数量，这样决策树
就会从全局出发学习拆分数据的最佳规则，但这样会导致决策树泛化能力强，但整体表现稍弱。为弥补这种方式的不足，可以构建多棵决策树，让他
们分别进行预测，再根据少数服从多数原则选出最终预测结果，这就是随机森林的原理。
随机森林的参数：
	#随机森林很多参数和决策树一样，但也引进了一些新参数
	(1)n_estimators: 指定创建决策树的数量
	(2)oob_score   : 如果设置为真，测试时将不使用训练模型时用过的数据
	(3)n_jobs      : 采用并行算法训练决策树时所用到的内核数量，默认为1，如果设置为-1，则启动CPU全部内核
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 14)
X_all = np.hstack([X_home_higher,X_teams])
scores = cross_val_score(clf,X_all,y,scoring = "accuracy")

使用 GridSearchCV类搜索最佳参数：
parameter_space = {
					"max_features":[2,10,'auto'],
					"n_estimators":[100,],
					"criterion":['gini','entropy'],
					"min_samples_leaf":[2,4,6],
					}
grid = GridSearchCV(clf,parameter_space)
grid.fit(X_all,y)
print grid.best_score_ * 100 #查看各种参数中的最高正确率
print grid.best_estimator_ # 查看最高正确率模型所用到的参数
#RandomForestClassifier(bootstrap=True, compute_importances=None,
#            criterion='entropy', max_depth=None, max_features=2,
#            max_leaf_nodes=None, min_density=None, min_samples_leaf=6,
#            min_samples_split=2, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=14, verbose=0)





************  numpy, pandas 使用  ************

1.读取csv数据
import pandas as pd
#读取数据，将数据保存在数据框(dataframe)中
dataset = pd.read_csv(data_filename)

2.输出数据集前5行
dataset.ix[:5]

3.数据集清洗
#数据日期是字符串格式，不是日期对象（Tue Oct 29 2013）
#有时候dataframe的第一行没有数据
#数据集的表头（columns）不完整
dataset = pd.read_csv(data_filename, parse_dates = ["Date"], skiprows = [0,]) #"Date"是那一列的名称,其内容是日期
dataset.columns = ["Date", "Score Type", "Visitor Team","VisitorPts", "Home Team", "HomePts"]

4.提取新特征，创建新的列
dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["HomeWin"].values #y_true的取值范围为[True,False]

5.对dataframe逐行访问
for index,row in dataset.iterrows():  #类似for index,num in enumerate(List A):

6.对dataframe排序
dataset.sort("Date")

7.对dataframe的某列进行计算
dataset["HomeWin"].sum() #计算HomeWin这一列的和，如果该列取值是布尔型，则计算True的个数
dataset["HomeWin"].count() #计算HomeWin这一列多少行，如果取值布尔型，.sum()/.count()计算了True值的比例



