import pandas as pd
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,6))
import numpy as np

data=pd.read_csv(r"...\data1.csv",header=None)

X=data.iloc[:,[0,1]]  #提取特征
y=data[2]   #提取目標

###把數據歸一化###
mean=X.mean(axis=0)
sigma=X.std(axis=0)
X=(X-mean)/sigma

###提取不同類別的數據，用於畫圖###
x_positive=X[y==1]
x_negative=X[y==-1]

ax.scatter(x_positive[0],x_positive[1],marker="o",label="y=+1")
ax.scatter(x_negative[0],x_negative[1],marker="x",label="y=-1")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Standardized Data")
ax.set_xlim(-2,2.6)
ax.set_ylim(-2,2.6)

X[2]=np.ones((X.shape[0],1))   #給特征增加一列常數項
X=X.values    #把特征轉換成ndarray格式

###初始化w###
w=X[0].copy()  #選取原點到第一個點的向量作為w的初始值
w[2]=0  #增加一項---閾值，閾值初始化為0
w=w.reshape(3,1)

y=y.values.reshape(100,1)  #把目標轉換成ndarray格式，形狀和預測目標相同
    
def compare(X,w,y):
    ###用於比較預測目標y_pred和實際目標y是否相符，返回分類錯誤的地方loc_wrong###
    ###輸入特征，權重，目標###
    scores=np.dot(X,w)  #把特征和權重點乘，得到此參數下預測出的目標分數
    
    y_pred=np.ones((scores.shape[0],1))  #設置預測目標，初始化值全為1，形狀和目標分數相同
    
    loc_negative=np.where(scores<0)[0]  #標記分數為負數的地方
    y_pred[loc_negative]=-1  #使標記為負數的地方預測目標變為-1
    
    loc_wrong=np.where(y_pred!=y)[0]  #標記分類錯誤的地方
    
    return loc_wrong

def update(X,w,y):
    ###用於更新權重w，返回更新後的權重w###
    ###輸入特征，權重，目標###
    w=w+y[compare(X,w,y)][0]*X[compare(X,w,y),:][0].reshape(3,1)
    return w

def perceptron(X,w,y):
    ###感知機算法，顯示最終的權重和分類直線，並畫出分類直線###
    ###輸入特征，初始權重，目標###
    while len(compare(X,w,y))>0:
        print("錯誤分類點有{}個。".format(len(compare(X,w,y))))
        w=update(X,w,y)

    print("參數w:{}".format(w))
    print("分類直線:{}x1+{}x2+{}=0".format(w[0][0],w[1][0],w[2][0]))
    line_x=np.linspace(-3,3,10)
    line_y=(-w[2]-w[0]*line_x)/w[1]
    ax.plot(line_x,line_y)

plt.show()




def perceptron_pocket(X,w,y):
    ###感知機口袋算法，顯示n次叠代後最好的權重和分類直線，並畫出分類直線###
    ###輸入特征，初始權重，目標###
    best_len=len(compare(X,w,y))  #初始化最少的分類錯誤點個數
    best_w=w  #初始化口袋裏最好的參數w
    for i in range(100):
        print("錯誤分類點有{}個。".format(len(compare(X,w,y))))
        w=update(X,w,y)
        #如果當前參數下分類錯誤點個數小於最少的分類錯誤點個數，那麽更新最少的分類錯誤點個數和口袋裏最好的參數w
        if len(compare(X,w,y))<best_len:
            best_len=len(compare(X,w,y))
            best_w=w

    print("參數best_w:{}".format(best_w))
    print("分類直線:{}x1+{}x2+{}=0".format(best_w[0][0],best_w[1][0],best_w[2][0]))
    print("最少分類錯誤點的個數:{}個".format(best_len))
    line_x=np.linspace(-3,3,10)
    line_y=(-best_w[2]-best_w[0]*line_x)/best_w[1]
    ax.plot(line_x,line_y)

def update(X,w,y):
    ###用於更新權重w，返回更新後的權重w###
    ###輸入特征，權重，目標###
    num=len(compare(X,w,y)) #分類錯誤點的個數
    w=w+y[compare(X,w,y)][np.random.choice(num)]*X[compare(X,w,y),:][np.random.choice(num)].reshape(3,1)
    return w

