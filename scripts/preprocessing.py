import sys
import pandas as pd

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

CANCER_NAMES = [ 'BRCA', 'COADREAD', 'GBM', 'LUAD', 'OV', 'UCEC' ]

# usage: python split_data.py <input> <fraction> <output_prefix>
# usage: python split_data.py <input> <count> <output_prefix>
input = sys.argv[1]
#fraction = float(sys.argv[2])
count = int(sys.argv[2])
output_prefix = sys.argv[3]

# load data
dataset = pd.read_csv(input)

# gene to index translate table
with open(output_prefix + "_gene.csv", "w") as gene_dataset:
    genes = dataset.groupby('Gene_Name').size().reset_index(name='Count')
    gene_index, gene_name = pd.factorize(genes['Gene_Name'])
    gene_dataset.write("Gene_Name,Index\n")
    for i in gene_index:
        gene = gene_name[i]
        gene_dataset.write("{},{}\n".format(gene,i))

# shuffle samples
samples = dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')
shuffle_samples = samples.sample(frac=1)

# split sample into train and test
sample_size = len(samples)
train_sample_size = int(sample_size * fraction)
train_sample = shuffle_samples[:train_sample_size]
test_sample = shuffle_samples[train_sample_size:]

# split datset into train and test
train_dataset = dataset[dataset['Tumor_Sample_ID'].isin(train_sample['Tumor_Sample_ID'])]
test_dataset = dataset[dataset['Tumor_Sample_ID'].isin(test_sample['Tumor_Sample_ID'])]

# train sample indexing
train_dataset = pd.unique(pd.read_csv(output_prefix+"/train.csv"))

with open(output_prefix + "_sample.csv", "w") as sample_dataset:
    sample_dataset.write("Tumor_Sample_ID,Tumor_Sample_Index\n")
    for i in range(len(train_sample)):
        line = "{},{}".format(train_sample['Tumor_Sample_ID'].iloc[i], i)
        sample_dataset.write(line+'\n')

# save train and test dataset
train_dataset.to_csv(output_prefix + "_train.csv", sep=',', index=False)
test_dataset.to_csv(output_prefix + "_test.csv", sep=',', index=False)
