import pandas as pd
import scipy.stats as st


column_names=['Subject_identifier', 'Jitter_2', 'Jitter_3', 'Jitter_4', 'Jitter_5', 'Jitter_6', 'Shimmer_7', 'Shimmer_8', 'Shimmer_9', 'Shimmer_10', 'Shimmer_11', 'Shimmer_12', 'Harmonicity_13', 'Harmonicity_14', 'Harmonocity_15', 'Pitch_16', 'Pitch_17', 'Pitch_18', 'Pitch_19', 'Pitch_20', 'Pulse_21', 'Pulse_22', 'Pulse_23', 'Pulse_24', 'Voice_25', 'Voice_26', 'Voice_27', 'UPDRS', 'PD_indicator']
# name columns of the dataset
df = pd.read_csv("po1_data.txt", names=column_names)


# group the dataset by calculating mean of each person's rows of data
grouped_df = df.groupby('Subject_identifier').mean()


# show information
print(grouped_df.head())


# split the dataset, 1 for PPD people and 1 for non PPD people
PPD = grouped_df.query('PD_indicator == 1') # suffer PPD
non_PPD = grouped_df.query('PD_indicator == 0') # not suffer PPD


# a different way of splitting 
# PPD = df[df["PD_indicator"]==1]
# non_PPD = df[df["PD_indicator"]==0]


# show information of 2 new datasets created 
print(PPD.head())

print(non_PPD.head())


# create 2 lists containing values of columns of PPD and non PPD datasets created above 
PPD_samples = []
non_PPD_samples = []
salient_features = []
count = 0

# loop through each feature (column) except "subject identifier", "PD indicator" and "UPDRS"
for column in column_names:
    if column in ["Subject_identifier", "PD_indicator", "UPDRS"]:
        continue
    PPD_samples = PPD[column].to_numpy()
    non_PPD_samples = non_PPD[column].to_numpy()

    # calculate means of the feature of 2 groups
    PPD_samples_mean = st.tmean(PPD_samples)
    non_PPD_samples_mean = st.tmean(non_PPD_samples)

    # calculate standard deviation values of the feature of 2 groups
    PPD_samples_std = st.tstd(PPD_samples)
    non_PPD_samples_std = st.tstd(non_PPD_samples)

    # calculate size of the feature of 2 groups
    PPD_samples_size = len(PPD_samples)
    non_PPD_samples_size = len(non_PPD_samples)

    # calculate p_value for this feature from 2 groups
    t_stats, p_value = st.ttest_ind_from_stats(PPD_samples_mean, PPD_samples_std, PPD_samples_size, 
                                                non_PPD_samples_mean, non_PPD_samples_std, non_PPD_samples_size,
                                                equal_var=False, alternative='two-sided')

    # check if the null hypothesis would be rejected
    if p_value < 0.05: # < 0.05 reject the null and this is the salient feature
        print(f"Column {column}: p-value = {p_value}")
        print(f"PPD: {PPD_samples_mean} and non_PPD: {non_PPD_samples_mean}\n")
        salient_features.append(column)
        count += 1
        


# print salient features and done
if count != 0:
    print("Set of salient acoustic features:")
    for feature in salient_features:
        print(feature)
else: 
    print("There's no salient acoustic features from the dataset to distinguish between PPD and non PPD")