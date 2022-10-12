import torch
import numpy as np

class K_center_greedy:
    def __init__(self, distance_matrix, 
                k_numb = None , 
                radius= 25.,
                initial_center_id="random"):

        self.weights = distance_matrix  #각 거리 NxN 메트릭스
        self.n = self.weights.shape[0]
        self.k = k_numb                         #how many centers
        self.dist = torch.Tensor(self.n)  #각 class와 center class 간의 거리 
        self.index_map = dict()                 #집합 요소 id와 그 요소의 center id  

        if initial_center_id=="random":
            self.initial_center_id = int(np.random.randint(self.n))  #초기 center class의 id 사용자가 커스터마이징 가능
        else:
            self.initial_center_id = initial_center_id
        self.centers = torch.Tensor([self.initial_center_id])        #center id list 초기화

        self.radius = radius               #클러스터의 최대 반지름
        self.select_error() #예외처리
        self.selectKcenter()

    def selectKcenter(self):
        #dist의 값을 모두 무한에 가깝게 만든다. 즉 첫 center의 탐색 범위를 무한으로 만든다.
        for i in range(self.n): 
            self.dist[i] = torch.Tensor([10**9])
            self.index_map.update({i: self.initial_center_id})

        i=0
        maxid = self.initial_center_id
        while True:
            for j in range(self.n):                             #center와 점들 간의 거리를 업데이트
                if self.weights[maxid][j] < self.dist[j]:
                    self.index_map[j] = maxid
                    self.dist[j] = self.weights[maxid][j]
                    
            maxid = int(torch.argmax(self.dist))
            self.centers = torch.cat((self.centers, torch.Tensor([maxid])), dim=0)
            i+=1
            
            if self.radius is not None  :                             #거리의 최대값을 지정한 경우
                if torch.max(self.dist) <= self.radius:
                    break
                
            if self.k is not None:
                if i== self.k:                                      #center 수의 값을 지정한 경우
                    break
        return [self.index_map, self.dist]                        # 종_id:center_id 딕셔너리,   id 마다 center로부터 거리

    def get_clusters(self, class_to_name):
        clusters = {}
        name_clusters = {}
        fine_to_coarse = {}
        center_to_coarse = {}
        
        for label, center in self.index_map.items():
            label_name = class_to_name[label]
            center_name = class_to_name[center]
            
            if center in clusters.keys():
                if not label in clusters[center]:
                    clusters[center].append(label)                    
                    name_clusters[center_name].append(label_name)
            else:
                clusters[center] = [label]
                name_clusters[center_name] = [label_name]
                
        for _class, _center in dict(sorted(self.index_map.items(), key=lambda item: item[1])).items():
            if _center not in center_to_coarse.keys():
                center_to_coarse[_center] = len(center_to_coarse)
            fine_to_coarse[_class] = center_to_coarse[_center]
                
        return clusters, name_clusters, fine_to_coarse, center_to_coarse
 
    def select_error(self): #select_k_center 함수의 예외처리
        #weight 행렬이 비어있는 경우
        if len(self.weights) ==0:
            raise Exception("Weights was not declared")
        elif self.weights.shape[0] != self.weights.shape[1]:
             raise Exception("Row and Column aren't match in Weights")

        #initial center id가 입력됐을 경우
        if type(self.initial_center_id) is not int:
            raise Exception("Please input int type initial center id")
        elif self.initial_center_id>= self.n or self.initial_center_id<0:
            raise Exception("initial center id must be bigger than 0 or smaller than {%d}",self.n)

        #k, radius 둘 중 하나는 무조건 Nonetype이여야함.
        if self.k is None: # 최대 반지름 지정
            if self.radius is None:
                raise Exception('Both k and radius are None.')
            elif (type(self.radius) is not int) and (type(self.radius) is not float):
                raise Exception('radius type error. now',type(self.radius))

        elif type(self.k) is not int:
            raise Exception('k type error. now',type(self.k))
        
        elif self.k == 0:
            raise Exception("k is 0. clustering is not possible. If you don't want to decide on k, change it to None.")
        
        else: # 최대 center 수 지정
            if self.radius is None:
                pass
            elif (type(self.radius) is not int) and (type(self.radius) is not float):
                raise Exception('radius type error. now',type(self.radius))
            else: #radius가 0 이상의 값을 가질 경우.
                raise Exception('k and radius are incompatible. Change one of the two to None')