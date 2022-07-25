from scipy.spatial import distance
import numpy as np


class Set_handle:

    def __init__(self,Set1, Set2):
        self.set1 = Set1
        self.set2 = Set2
        self.distance = None




    def mean_covariance(self,conv_set):
        mean_vec = np.mean(conv_set, axis=0)
        covar_matrix = np.cov(conv_set.T)
        return mean_vec, covar_matrix


    
    def hellinger_distance(self): #헬링거 거리
        self.set_error()

        mean1, cov_matrix1 = self.mean_covariance(self.set1)
        mean2, cov_matrix2 = self.mean_covariance(self.set2)
        avg_cov = (cov_matrix1 + cov_matrix2)/2
        inv_cov = nl.matrix_power(avg_cov,-1)

        hell_dist =  np.sqrt(np.sqrt(nl.det(cov_matrix1))) * np.sqrt(np.sqrt(nl.det(cov_matrix2)))/ np.sqrt(nl.det(avg_cov))
        hell_dist *= np.exp(-(mean1- mean2).T.dot(inv_cov).dot(mean1- mean2)/8)
        hell_dist = np.sqrt(1- hell_dist)
        return hell_dist

    def bhattacharyya_distance(self): #바타차리야 거리
        self.set_error()

        mean1, cov_matrix1 = self.mean_covariance(self.set1)
        mean2, cov_matrix2 = self.mean_covariance(self.set2)
        avg_cov = (cov_matrix1 + cov_matrix2)/2
        inv_cov = nl.matrix_power(avg_cov,-1)

        bha_dist = (mean1- mean2).T.dot(inv_cov).dot(mean1- mean2)/8
        bha_dist += ln(nl.det(avg_cov)/ np.sqrt(nl.det(cov_matrix1) * nl.det(cov_matrix2)))/2
        return bha_dist

    def kullback_leibler_divergence(self): #쿨백-리버 발산
        self.set_error()

        mean1, cov_matrix1 = self.mean_covariance(self.set1)
        mean2, cov_matrix2 = self.mean_covariance(self.set2)
        inv_cov2 = nl.matrix_power(cov_matrix2,-1)
        k= self.set1.shape[1]

        kl_dist = (np.trace(np.dot(inv_cov2,cov_matrix1)) - k + (mean2- mean1).T.dot(inv_cov2).dot(mean2- mean1) + ln(nl.det(cov_matrix2)/nl.det(cov_matrix1)) )/2
        return kl_dist

    def set_error(self):
        if (type(self.set1) is not np.ndarray) or type(self.set2) is not np.ndarray:
            raise Exception("Vector sets invaild type. it have to numpy.ndarray")
        elif self.set1.shape[1] != self.set2.shape[1]:
            raise Exception("Shape of two vector sets does not fit")
    

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

    def distance_matrix(self, method="hellinger_distance"):
        print("현재 클러스터링 거리 함수는 "+method)
        if method == "bhattacharyya_distance":
            self.weights = []
            for idx in range(self.n):
                std_vector_set = self.vector_set[idx]
                dist=[]
                for jdx in range(self.n):
                    sd = Set_handle(Set1 = std_vector_set, Set2= self.vector_set[jdx]).bhattacharyya_distance()
                    dist.append(sd)
                
                self.weights.append(dist)
            self.weights = np.array(self.weights)
            print(self.weights.shape)
            return self.weights

        elif method == "hellinger_distance":
            self.weights = []
            for idx in range(self.n):
                std_vector_set = self.vector_set[idx]
                dist=[]
                for jdx in range(self.n):
                    sd = Set_handle(Set1 = std_vector_set, Set2= self.vector_set[jdx]).hellinger_distance()
                    dist.append(sd)
                
                self.weights.append(dist)
            self.weights = np.array(self.weights)
            print(self.weights.shape)
            return self.weights

        elif method == "kullback_leibler_divergence":
            self.weights = []
            for idx in range(self.n):
                std_vector_set = self.vector_set[idx]
                dist=[]
                for jdx in range(self.n):
                    sd = Set_handle(Set1 = std_vector_set, Set2= self.vector_set[jdx]).kullback_leibler_divergence()
                    dist.append(sd)
                
                self.weights.append(dist)
            self.weights = np.array(self.weights)
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
