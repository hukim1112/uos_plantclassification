from scipy.spatial import distance
import numpy as np


class Set_distance:

    def __init__(self,Set1, Set2):
        self.set1 = Set1
        self.set2 = Set2
        self.distance = None

        if (type(self.set1) is not np.ndarray) or type(self.set2) is not np.ndarray:
            raise Exception("Vector sets invaild type. it have to numpy.ndarray")


    def L_matrix(self, L=2):
        Row = self.set1.shape[0]
        Column = self.set2.shape[0]
        out = [  [ np.linalg.norm((self.set1[i]-self.set2[j]),L)   for i in range(Row) ] for j in range(Column) ]
        return np.array(out)

    def mean_covariance_matrix(self, Vector_set):
        set_mean = np.mean(Vector_set, axis=0)
        return 0

    def simple_max_distance(self,L=2):      
        return np.max(self.L_matrix(L))
    

class Kcenter:

    def __init__(self, vectors):
        self.vector_set = vectors
        self.n = len(self.vector_set)

        self.weights = np.array([]) #각 거리 NxN 메트릭스

        self.k = None                              #how many centers
        self.centers =[]                        #center id들
        self.dist =[0] * self.n                 #dist는 집합 요소와 center 간의 거리
        self.index_map = dict()                 #집합 요소 id와 그 요소의 center id 
        self.maxid = np.random.randint(self.n)  #처음 center의 id 사용자가 커스터마이징 가능
        self.threshold =None

    def maxindex(self,dist):
        mi = 0
        for i in range(self.n): 
            if (dist[i] > dist[mi]):
                mi = i
        return mi  #dist 벡터 중 가장 큰 값을 가진 index를 찾는다.

    def distance_matrix(self, method="simple_max_distance"):
        if method == "simple_max_distance":

            for idx in range(self.n):
                std_vector_set = self.vector_set[idx]
                dist=[]
                for jdx in range(self.n):
                    SD = Set_distance(Set1 = std_vector_set, Set2= self.vector_set[jdx])
                    dist.append(SD.simple_max_distance())
                print(self.weights.shape)
                
                self.weights= np.append(self.weights, dist, axis=0)
                
            print(self.weights.shape)
            return self.weights
        else:
            raise Exception("please choose other method")


    def selectKcenter(self, threshold= None):
        self.threshold = threshold
        self.select_error() #예외처리

        #dist의 값을 모두 무한에 가깝게 만든다. 즉 첫 center의 탐색 범위를 무한으로 만든다.
        for i in range(self.n): 
            self.dist[i] = 10 ** 9
            self.index_map.update({i: self.maxid})
        
        i=0
        while True:
            self.centers.append(self.maxid)
            for j in range(self.n):                             #center와 점들 간의 거리를 업데이트
                if self.weights[self.maxid][j] < self.dist[j]:
                    self.index_map[j] = self.maxid
                    self.dist[j] = self.weights[self.maxid][j]

            self.maxid = self.maxindex(self.dist)

            i+=1
            if threshold != None  :                             #거리의 최대값을 지정한 경우
                if self.dist[self.maxid] <= threshold:
                    break
            
            if i== self.k:                                      #center 수의 값을 지정한 경우
                break

        for i in self.centers:
            print(i, end=" ")

        return self.index_map, self.dist                        # 종_id:center_id 딕셔너리,   id 마다 center로부터 거리



    
        




    def select_error(self): #select_k_center 함수의 예외처리
        #weight 행렬이 비어있는 경우
        if len(self.weights) ==0:
            raise Exception("Weights was not declared")
        elif self.weights.shape[0] != self.weights.shape[1]:
             raise Exception("Row and Column aren't match in Weights")

        #k, threshold 둘 중 하나는 무조건 Nonetype이여야함.
        if self.k is None: # 최대 반지를 지정
            if self.threshold is None:
                raise Exception('Both k and threshold are None.')
            elif (type(self.threshold) is not int) and (type(self.threshold) is not float):
                raise Exception('threshold type error. now',type(self.threshold))

        elif type(self.k) is not int:
            raise Exception('k type error. now',type(self.k))
        
        elif self.k == 0:
            raise Exception("k is 0. clustering is not possible. If you don't want to decide on k, change it to None.")
        
        else: # 최대 center 수 지정
            if self.threshold is None:
                pass
            elif (type(self.threshold) is not int) and (type(self.threshold) is not float):
                raise Exception('threshold type error. now',type(self.threshold))
            else: #threshold가 0 이상의 값을 가질 경우.
                raise Exception('k and threshold are incompatible. Change one of the two to None')