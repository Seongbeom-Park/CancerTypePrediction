import sys
import pandas as pd

# usage: python split_data.py <input> <fraction> <output_prefix>
input = sys.argv[1]
fraction = float(sys.argv[2])
output_prefix = sys.argv[3]

# load data
dataset = pd.read_csv(input)
samples = dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Mutation_Count')
sample_size = len(samples)
train_sample_size = int(sample_size * fraction)

# shuffle samples
shuffle_samples = samples.sample(frac=1)

# split sample into train and test
train_sample = shuffle_samples[:train_sample_size]
test_sample = shuffle_samples[train_sample_size:]

# split datset into train and test
train_dataset = dataset[dataset['Tumor_Sample_ID'].isin(train_sample['Tumor_Sample_ID'])]
test_dataset = dataset[dataset['Tumor_Sample_ID'].isin(test_sample['Tumor_Sample_ID'])]

# save train and test dataset
train_dataset.to_csv(output_prefix + "_train.csv", sep=',')
test_dataset.to_csv(output_prefix + "_test.csv", sep=',')
