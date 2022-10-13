from experiment.training import train_EfficientB4, train_ResNet101, train_VGG19, train_WideResNet101_2, train_hierarchical_EfficientB4, train_genera_species_hierarchical_classifier

train_genera_species_hierarchical_classifier("cuda:0", "finetune")