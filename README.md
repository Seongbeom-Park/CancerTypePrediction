# CancerTypePrediction
cse611 BIO machine learning project

## Prepare input file
$ python scripts/preprocessing.py 0.8 TCGA_6_Cancer_Type_Mutation_List.csv

$ mkdir TCGA_6_Cancer_Type_Mutation_List

$ mv TCGA_6_Cancer_Type_Mutation_List_sample.csv TCGA_6_Cancer_Type_Mutation_List/sample.csv

$ mv TCGA_6_Cancer_Type_Mutation_List_cancer.csv TCGA_6_Cancer_Type_Mutation_List/cancer.csv

$ mv TCGA_6_Cancer_Type_Mutation_List_gene.csv TCGA_6_Cancer_Type_Mutation_List/gene.csv

$ mv TCGA_6_Cancer_Type_Mutation_List_train.csv TCGA_6_Cancer_Type_Mutation_List/train.csv

$ mv TCGA_6_Cancer_Type_Mutation_List_test.csv TCGA_6_Cancer_Type_Mutation_List/test.csv


## Run scripts

$ python scripts/cancer_nn.py TCGA_6_Cancer_Type_Mutation_List
