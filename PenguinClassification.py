import csv
import os
import copy
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class GetCSV:
    dirName = os.path.dirname(__file__)
    def __init__(self,fileName,features,mapping):
        self.fileName = os.path.join(self.dirName,fileName)
        self.data = []
        #csvファイルの読み込み
        with open(self.fileName,newline="") as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]
        #先頭行を取り除く
        self.firstRow = data.pop(0)
        indexes = []
        #取得する列名から列番号を取得
        for index in features:
            indexes.append(self.firstRow.index(index))
        for row in data:
            #データの欠損がない行に対して取得処理を行う
            if row.count('') == 0:
                tmprow = []
                for index in indexes:
                    #列にある文字列が辞書型リストに登録されている場合、数値データに置き換える
                    if row[index] in mapping :
                        row[index] = mapping[row[index]]
                    tmprow.append(row[index])
                self.data.append(tmprow)
    #取得したデータを返すメソッド
    def getData(self):
        return self.data
    #先頭行を返すメソッド
    def getFirstRow(self):
        return self.firstRow
    #目的変数となるデータのリストを返す
    def getDependent(self,label):
        index = self.firstRow.index(label)
        tmpdata = []
        data = copy.deepcopy(self.data)
        for row in data:
            tmpdata.append(row.pop(index))
        return tmpdata
    #説明変数となるデータのリストを返す
    def getExplanatory(self,label):
        index = self.firstRow.index(label)
        tmpdata = []
        data = copy.deepcopy(self.data)
        for row in data:
            row.pop(index)
            tmpdata.append(row)
        return tmpdata


class Kmeans:
    def __init__(self,n_clusters,data):
        self.n_clusters = n_clusters
        self.data = data
        self.kmeans = KMeans(self.n_clusters).fit(self.data)

    #分類結果を返す
    def getLabels(self):
        return self.kmeans.labels_
    
    #分類結果を標準出力に出す
    def printLabels(self):
        print(self.getLabels())
    
class SVM:
    def __init__(self,explanatory,dependent):
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(explanatory,dependent,test_size=0.3,random_state=None)
        self.model = SVC()
        self.model.fit(self.x_train,self.y_train)
    
    #トレーニングデータに対する正答率を標準出力に出す
    def printTrainScore(self):
        predTrain = self.model.predict(self.x_train)
        accuracyTrain = accuracy_score(self.y_train,predTrain)
        print("トレーニングデータに対する正答率：%.2f" % accuracyTrain)

    #テストデータに対する正答率を標準出力に出す
    def printTestScore(self):
        predTest = self.model.predict(self.x_test)
        accuracyTest = accuracy_score(self.y_test,predTest)
        print("テストデータに対する正答率：%.2f" % accuracyTest)

    
if __name__ == "__main__":
    ###############
    #設定値
    ###############
    #学習元データのファイル名
    fileName = "penguins.csv"
    #ファイルから取得する列名
    features = ["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
    #目的変数の列名
    label = "species"
    #学習元データの分類が文字列なので、辞書型リストで数値に変換する
    mapping = {"Adelie":0,"Chinstrap":1,"Gentoo":2}
    #k-meansにおける分類数
    clusters = 3
    ###############
    csv = GetCSV(fileName=fileName,features=features,mapping=mapping)
    kmeans = Kmeans(n_clusters=clusters,data=csv.getData())
    svm = SVM(csv.getExplanatory(label),csv.getDependent(label))
    print("k-meansによる分類結果")
    kmeans.printLabels()
    print("SVMを用いた教師あり学習")
    svm.printTrainScore()
    svm.printTestScore()
    