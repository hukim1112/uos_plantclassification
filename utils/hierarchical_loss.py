import torch
import torch.nn as nn

class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, fine_to_coarse, device, total_level=2, alpha=1, beta=0.8, p_loss=3):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.step_count = 0
        self.p_loss = p_loss
        self.fine_to_coarse = fine_to_coarse
        self.device = device

    def __call__(self, predictions, true_labels):
        self.step_count += 1
        if self.beta == "scheduler":
            self.beta_scheduler()
        dloss = self.calculate_dloss(predictions, true_labels)
        lloss = self.calculate_lloss(predictions, true_labels)
        total_loss = lloss + dloss
        return total_loss
    
    def beta_scheduler(self):
        if self.step_count<501:
            self.beta = 0.0
        elif self.step_count <1001:
            self.beta = 0.2
        elif self.step_count <1501:
            self.beta = 0.4        
        elif self.step_count <2001:
            self.beta = 0.6        
        else:
            self.beta = 0.8
                
    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss

    def check_hierarchy(self, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''        
        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [ not(previous_level[i].item() == self.fine_to_coarse[str(current_level[i].item())]) for i in range(current_level.size()[0])]
        
        return torch.FloatTensor(bool_tensor).to(self.device)

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred)

            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
    
class MHLN(HierarchicalLossNetwork):
    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''
        lloss = 0
        for l in range(self.total_level):
            lloss += self.focal_loss(predictions[l], true_labels[l])
        return self.alpha * lloss
    
class NO_HC_DEPENDENCY(HierarchicalLossNetwork):
    def __call__(self, predictions, true_labels):
        self.step_count += 1
        if self.beta == "scheduler":
            self.beta_scheduler()
        #dloss = self.calculate_dloss(predictions, true_labels)
        lloss = self.calculate_lloss(predictions, true_labels)
        total_loss = lloss
        return total_loss