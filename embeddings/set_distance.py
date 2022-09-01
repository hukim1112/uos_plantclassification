import torch
import numpy as np

class SetDist:  #Set1, Set2 모두 tensor
    def __init__(self, dist_mode="bhattacharyya_distance"):
        self.dist_mode = "bhattacharyya_distance"

    def __call__(self, set1, set2):
        if self.dist_mode == "bhattacharyya_distance":
            return self.bhattacharyya_distance(set1,set2)
        elif self.dist_mode == "kullback_leibler":
            return self.kullback_leibler_divergence(set1, set2)
        else:
            raise ValueError(f"wrong dist mode {self.dist_mode}")
            
    def set_error(self, set1, set2):
        if (type(set1) is not torch.Tensor) and (type(set2) is not torch.Tensor):
            raise Exception("Vector sets invaild type. it have to torch.Tensor")
        elif set1.shape[1] != set2.shape[1]:
            raise Exception("Shape of two vector sets does not fit")
    
    def log_det(self,cov):
        l = torch.linalg.cholesky(cov) #하삼각행렬과 그 행렬의 대칭행렬로 cov를 분해
        logdiagL = 2*torch.sum(torch.log(torch.diag(l, 0)))
        return logdiagL

    def multi_variate_gaussian(self,vector_set):
        # calculate multivariate Gaussian distribution
        B, C = vector_set.size()
        mean = torch.mean(vector_set, dim=0)
        cov = torch.zeros(C, C)
        I = torch.eye(C)

        # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
        cov= torch.cov(vector_set.T) + 0.01 * I

        inv_cov = torch.linalg.inv(cov)

        return [mean, cov, inv_cov]

    def bhattacharyya_distance(self, set1, set2): #바타차리야 거리
        self.set_error(set1, set2)
        mean1, cov_matrix1, _ = self.multi_variate_gaussian(set1)  #1. 평균 벡터, 2. 공분산 행렬,  3. 공분산 행렬의 역행렬
        mean2, cov_matrix2, _ = self.multi_variate_gaussian(set2)
        avg_cov = (cov_matrix1 + cov_matrix2)/2
        inv_cov = torch.linalg.inv(avg_cov)

        delta = (mean1 - mean2)
        delta1 =torch.matmul(delta.T, inv_cov).T
        delta2 = torch.matmul( delta1, delta)/8
        dets = (self.log_det(avg_cov) - (self.log_det(cov_matrix1) + self.log_det(cov_matrix2))/2)/2
        return delta2 + dets

    def bhattacharyya_from_stats(self, stat1, stat2):
        mean1, cov_matrix1, _ = stat1
        mean2, cov_matrix2, _ = stat2
        avg_cov = (cov_matrix1 + cov_matrix2)/2
        inv_cov = torch.linalg.inv(avg_cov)

        delta = (mean1 - mean2)
        delta1 =torch.matmul(delta.T, inv_cov).T
        delta2 = torch.matmul( delta1, delta)/8
        dets = (self.log_det(avg_cov) - (self.log_det(cov_matrix1) + self.log_det(cov_matrix2))/2)/2
        return delta2 + dets

    def mahalanobis_distance(self, embedding, stats):
        mean, cov, inv_covariance = stats
        delta = (embedding - mean)

        distance = (torch.matmul(delta, inv_covariance) * delta).sum()
        distance = torch.sqrt(distance)
        return distance

    def kullback_leibler_divergence(self, set1, set2): #쿨백-리버 발산
        self.set_error(set1, set2)
        mean0, cov_matrix0,_ = self.multi_variate_gaussian(set1)
        mean1, cov_matrix1, _ = self.multi_variate_gaussian(set2)
        inv_cov1 = torch.linalg.inv(cov_matrix1)
        k= self.set1.shape[1]

        delta = (mean1 - mean0)
        delta1 =torch.matmul(delta.T, inv_cov1).T
        delta2 = torch.matmul( delta1, delta)

        kl_dist1 = torch.trace(torch.matmul(inv_cov1,cov_matrix0)) - k +delta2
        kl_dist2 = self.log_det(cov_matrix1) - self.log_det(cov_matrix0)  #공분산 행렬의 det 값의 비율을 구함, Decimal type을 tensor type으로 바꿉니다.

        return (kl_dist1+kl_dist2)/2


