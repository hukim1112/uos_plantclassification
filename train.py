from experiment.training import train_EfficientB4, train_ResNet101, train_WideResNet101_2, train_VGG19, train_genera_species_hierarchical_classifier

train_EfficientB4("cuda:2")

#train_genera_species_hierarchical_classifier("cuda:2", "fixed_extractor")