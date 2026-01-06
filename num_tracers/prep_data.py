import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

home_dir = os.environ["HOME"]

# set to a new version when updating the data
data_version = 4
data_release = 2
os.makedirs(os.path.join(home_dir, 'data/desi/tracers_v' + str(data_version)), exist_ok=True)

if data_release == 1:

    # efficiencies from this paper: https://arxiv.org/pdf/2411.12020 (Table 2)
    eff_BGS = 0.989
    eff_LRG = 0.991
    # assume the VLO efficiency is 95% and the LOP efficiency is 70% (https://arxiv.org/pdf/2306.06307)
    #eff_ELG_VLO = 0.95
    #eff_ELG_LOP = 0.7
    eff_ELG = 0.727
    eff_QSO = 0.668
    eff_LyaQSO = 0.668

    comp_BGS = 0.636
    comp_LRG = 0.693
    comp_ELG = 0.352
    comp_QSO = 0.874
    comp_LyaQSO = 0.874

    # from table 1 (DR1 paper)
    passed_BGS = 300017
    passed_LRG1 = 506911
    passed_LRG2 = 771894
    passed_LRG3 = 859822
    passed_ELG1 = 1016365
    passed_ELG2 = 1415707
    passed_QSO = 856652
    passed_LyaQSO = 709565

    # from table 1 (DR1 paper)
    zeff_BGS = 0.295
    zeff_LRG1 = 0.510
    zeff_LRG2 = 0.706
    zeff_LRG3ELG1 = 0.930
    zeff_ELG2 = 1.317
    zeff_QSO = 1.491
    zeff_LyaQSO = 2.330

elif data_release == 2:
    # All data is gathered from tables in the DR2 paper (https://arxiv.org/html/2503.14738v1)

    # from table 2
    eff_BGS = 0.988
    eff_LRG = 0.990
    eff_ELG = 0.739
    eff_QSO = 0.680
    eff_LyaQSO = 0.680

    # from table 2
    comp_BGS = 0.755
    comp_LRG = 0.826
    comp_ELG = 0.537
    comp_QSO = 0.936
    comp_LyaQSO = 0.936

    # from table 3
    passed_BGS = 1188526
    passed_LRG1 = 1052151
    passed_LRG2 = 1613562
    passed_LRG3 = 1802770
    passed_ELG1 = 2737573
    passed_ELG2 = 3797271
    passed_QSO = 1461588
    passed_LyaQSO = 1289874

    # from table 4
    zeff_BGS = 0.295
    zeff_LRG1 = 0.510
    zeff_LRG2 = 0.706
    zeff_LRG3ELG1 = 0.934
    zeff_ELG2 = 1.321
    zeff_QSO = 1.484
    zeff_LyaQSO = 2.330

obs_BGS = passed_BGS / eff_BGS
obs_LRG1 = passed_LRG1 / eff_LRG
obs_LRG2 = passed_LRG2 / eff_LRG
obs_LRG3 = passed_LRG3 / eff_LRG
obs_ELG1 = passed_ELG1 / eff_ELG
obs_ELG2 = passed_ELG2 / eff_ELG
obs_QSO = passed_QSO / eff_QSO
obs_LyaQSO = passed_LyaQSO / eff_QSO
obs_LRG3ELG1 = obs_LRG3 + obs_ELG1

passed_LRG3ELG1 = passed_LRG3 + passed_ELG1
eff_LRG3ELG1 = (obs_LRG3 / obs_LRG3ELG1) * eff_LRG + (obs_ELG1 / obs_LRG3ELG1) * eff_ELG
obs_LRG3ELG1 = passed_LRG3ELG1 / eff_LRG3ELG1

#desi_data = '/home/ashandonay/cobaya/packages/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
#cov_matrix = np.loadtxt('/home/ashandonay/cobaya/packages/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')
desi_data = home_dir + f'/data/desi/bao_dr{data_release}/desi_gaussian_bao_ALL_GCcomb_mean.txt'
cov_matrix = np.loadtxt(home_dir + f'/data/desi/bao_dr{data_release}/desi_gaussian_bao_ALL_GCcomb_cov.txt')

column_names = ['z', 'value_at_z', 'quantity']
desi_df = pd.read_csv(desi_data, delimiter=' ', names=column_names, comment='#')

cov_matrix = cov_matrix[np.ix_(desi_df.index, desi_df.index)]
corr_matrix = cov_matrix/np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
cov_path = os.path.join(home_dir, 'data/desi/tracers_v' + str(data_version), 'desi_cov.npy')
corr_path = os.path.join(home_dir, 'data/desi/tracers_v' + str(data_version), 'desi_corr.npy')
print(f"Saving cov matrix to {cov_path}")
np.save(cov_path, cov_matrix)
print(f"Saving corr matrix to {corr_path}")
np.save(corr_path, corr_matrix)

desi_df.insert(0, 'tracer', '')
desi_df.insert(1, 'passed', 0.0)
desi_df.insert(2, 'observed', 0.0)
desi_df.insert(4, 'efficiency', 1.0)
desi_df.insert(5, 'std', 0.0)
# add the diagonal of the cov matrix to the dataframe
desi_df.loc[:, 'std'] = np.sqrt(np.diag(cov_matrix))

desi_df.loc[desi_df['z'] == zeff_BGS, ["tracer", "passed", "observed"]] = "BGS", passed_BGS, obs_BGS
desi_df.loc[desi_df['z'] == zeff_LRG1, ["tracer", "passed", "observed"]] = "LRG1", passed_LRG1, obs_LRG1
desi_df.loc[desi_df['z'] == zeff_LRG2, ["tracer", "passed", "observed"]] = "LRG2", passed_LRG2, obs_LRG2
desi_df.loc[desi_df['z'] == zeff_LRG3ELG1, ["tracer", "passed", "observed"]] = "LRG3+ELG1", passed_LRG3ELG1, obs_LRG3ELG1
desi_df.loc[desi_df['z'] == zeff_ELG2, ["tracer", "passed", "observed"]] = "ELG2", passed_ELG2, obs_ELG2
desi_df.loc[desi_df['z'] == zeff_QSO, ["tracer", "passed", "observed"]] = "QSO", passed_QSO, obs_QSO
desi_df.loc[desi_df['z'] == zeff_LyaQSO, ["tracer", "passed", "observed"]] = "Lya QSO", passed_LyaQSO, obs_LyaQSO


"""
# read a text file in as a pandas dataframe
ELG_NGC_nz = pd.read_table('/home/ashandonay/data/ELGnotqso_NGC_nz.txt', delimiter=' ', names=['zmid', 'zlow', 'zhigh', 'n(z)', 'Nbin', 'Vol_bin'], comment='#')
ELG_SGC_nz = pd.read_table('/home/ashandonay/data/ELGnotqso_SGC_nz.txt', delimiter=' ', names=['zmid', 'zlow', 'zhigh', 'n(z)', 'Nbin', 'Vol_bin'], comment='#')
LOP_NGC_nz = pd.read_table('/home/ashandonay/data/ELG_LOPnotqso_NGC_nz.txt', delimiter=' ', names=['zmid', 'zlow', 'zhigh', 'n(z)', 'Nbin', 'Vol_bin'], comment='#')
LOP_SGC_nz = pd.read_table('/home/ashandonay/data/ELG_LOPnotqso_SGC_nz.txt', delimiter=' ', names=['zmid', 'zlow', 'zhigh', 'n(z)', 'Nbin', 'Vol_bin'], comment='#')
redshift_bins = ELG_NGC_nz['zmid'].values

# Combine the NGC and SGC data and separate the ELG and LOP tracers
ELG_nz = ELG_NGC_nz.copy()
ELG_nz['n(z)'] = ELG_NGC_nz['n(z)'] + ELG_SGC_nz['n(z)']
ELG_nz['Nbin'] = ELG_NGC_nz['Nbin'] + ELG_SGC_nz['Nbin']
ELG_nz['Vol_bin'] = ELG_NGC_nz['Vol_bin'] + ELG_SGC_nz['Vol_bin']

VLO_NGC_nz = ELG_NGC_nz.copy()
VLO_NGC_nz['n(z)'] = ELG_NGC_nz['n(z)'] - LOP_NGC_nz['n(z)']
VLO_NGC_nz['Nbin'] = ELG_NGC_nz['Nbin'] - LOP_NGC_nz['Nbin']
VLO_NGC_nz['Vol_bin'] = ELG_NGC_nz['Vol_bin'] - LOP_NGC_nz['Vol_bin']
VLO_SGC_nz = ELG_SGC_nz.copy()
VLO_SGC_nz['n(z)'] = ELG_SGC_nz['n(z)'] - LOP_SGC_nz['n(z)']
VLO_SGC_nz['Nbin'] = ELG_SGC_nz['Nbin'] - LOP_SGC_nz['Nbin']
VLO_SGC_nz['Vol_bin'] = ELG_SGC_nz['Vol_bin'] - LOP_SGC_nz['Vol_bin']

VLO_nz = VLO_NGC_nz.copy()
VLO_nz['n(z)'] = VLO_NGC_nz['n(z)'] + VLO_SGC_nz['n(z)']
VLO_nz['Nbin'] = VLO_NGC_nz['Nbin'] + VLO_SGC_nz['Nbin']
VLO_nz['Vol_bin'] = VLO_NGC_nz['Vol_bin'] + VLO_SGC_nz['Vol_bin']
LOP_nz = LOP_NGC_nz.copy()
LOP_nz['n(z)'] = LOP_NGC_nz['n(z)'] + LOP_SGC_nz['n(z)']
LOP_nz['Nbin'] = LOP_NGC_nz['Nbin'] + LOP_SGC_nz['Nbin']
LOP_nz['Vol_bin'] = LOP_NGC_nz['Vol_bin'] + LOP_SGC_nz['Vol_bin']

ELG1_VLO = np.trapz(np.array(VLO_nz.loc[(VLO_nz['zmid'] >= 0.8) & (VLO_nz['zmid'] <= 1.1), 'n(z)'].values), x=VLO_nz.loc[(VLO_nz['zmid'] >= 0.8) & (VLO_nz['zmid'] <= 1.1), 'zmid'].values)
ELG2_VLO = np.trapz(np.array(VLO_nz.loc[(VLO_nz['zmid'] >= 1.1) & (VLO_nz['zmid'] <= 1.6), 'n(z)'].values), x=VLO_nz.loc[(VLO_nz['zmid'] >= 1.1) & (VLO_nz['zmid'] <= 1.6), 'zmid'].values)
ELG1_LOP = np.trapz(np.array(LOP_nz.loc[(LOP_nz['zmid'] >= 0.8) & (LOP_nz['zmid'] <= 1.1), 'n(z)'].values), x=LOP_nz.loc[(LOP_nz['zmid'] >= 0.8) & (LOP_nz['zmid'] <= 1.1), 'zmid'].values)
ELG2_LOP = np.trapz(np.array(LOP_nz.loc[(LOP_nz['zmid'] >= 1.1) & (LOP_nz['zmid'] <= 1.6), 'n(z)'].values), x=LOP_nz.loc[(LOP_nz['zmid'] >= 1.1) & (LOP_nz['zmid'] <= 1.6), 'zmid'].values)

eff_comb_ELG1 = (ELG1_VLO / (ELG1_VLO + ELG1_LOP)) * eff_ELG_VLO + (ELG1_LOP / (ELG1_VLO + ELG1_LOP)) * eff_ELG_LOP
eff_comb_ELG2 = (ELG2_VLO / (ELG2_VLO + ELG2_LOP)) * eff_ELG_VLO + (ELG2_LOP / (ELG2_VLO + ELG2_LOP)) * eff_ELG_LOP

#num_ELG2 = desi_df.loc[desi_df["tracer"] == "ELG2", "num"].iloc[0]
#num_ELG1 = int(((ELG1_VLO + ELG1_LOP) / (ELG2_VLO + ELG2_LOP)) * num_ELG2)
#num_LRG3ELG1 = desi_df.loc[desi_df["tracer"] == "LRG3+ELG1", "num"].iloc[0]
#num_LRG3 = num_LRG3ELG1 - num_ELG1

"""

desi_df.loc[desi_df['z'] == zeff_BGS, "efficiency"] = eff_BGS
desi_df.loc[desi_df['z'] == zeff_LRG1, "efficiency"] =  eff_LRG
desi_df.loc[desi_df['z'] == zeff_LRG2, "efficiency"] = eff_LRG
desi_df.loc[desi_df['z'] == zeff_LRG3ELG1, "efficiency"] = eff_LRG3ELG1
desi_df.loc[desi_df['z'] == zeff_ELG2, "efficiency"] = eff_ELG
#desi_df.loc[desi_df['z'] == 0.930, "efficiency"] = (num_ELG1 / num_LRG3ELG1) * ((ELG1_VLO / (ELG1_VLO + ELG1_LOP)) * eff_ELG_VLO + (ELG1_LOP / (ELG1_VLO + ELG1_LOP)) * eff_ELG_LOP) + (1 - (num_ELG1 / num_LRG3ELG1)) * eff_LRG
#desi_df.loc[desi_df['z'] == 1.317, "efficiency"] = (ELG2_VLO / (ELG2_VLO + ELG2_LOP)) * eff_ELG_VLO + (ELG2_LOP / (ELG2_VLO + ELG2_LOP)) * eff_ELG_LOP
desi_df.loc[desi_df['z'] == zeff_QSO, "efficiency"] = eff_QSO
desi_df.loc[desi_df['z'] == zeff_LyaQSO, "efficiency"] = eff_LyaQSO
data_path = os.path.join(home_dir, 'data/desi/tracers_v' + str(data_version), 'desi_data.csv')
print(f"Saving data to {data_path}")
desi_df.to_csv(data_path, index=False)

desi_tracers = pd.DataFrame({
    'tracer': ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG1', 'ELG2', 'QSO', 'Lya QSO'],
    'class': ['BGS', 'LRG', 'LRG', 'LRG', 'ELG', 'ELG', 'QSO', 'QSO'],
    'efficiency': [eff_BGS, eff_LRG, eff_LRG, eff_LRG, eff_ELG, eff_ELG, eff_QSO, eff_LyaQSO],
    'comp': [comp_BGS, comp_LRG, comp_LRG, comp_LRG, comp_ELG, comp_ELG, comp_QSO, comp_LyaQSO],
    'passed': [passed_BGS, passed_LRG1, passed_LRG2, passed_LRG3, passed_ELG1, passed_ELG2, passed_QSO, passed_LyaQSO],
    })
desi_tracers['observed'] = desi_tracers['passed'] / desi_tracers['efficiency']
total_observations = desi_tracers['observed'].sum()
total_passed = desi_tracers['passed'].sum()
print(f"Total observations: {total_observations}, Total passed: {total_passed}")
desi_tracers['targets'] = desi_tracers['observed'] / desi_tracers['comp']

# normalize the passed and observed counts by the total number of tracers
desi_tracers['passed'] = desi_tracers['passed'] / total_passed
desi_tracers['observed'] = desi_tracers['observed'] / total_observations

tracers_path = os.path.join(home_dir, 'data/desi/tracers_v' + str(data_version), 'desi_tracers.csv')
print(f"Saving tracers to {tracers_path}")
desi_tracers.to_csv(tracers_path, index=False)