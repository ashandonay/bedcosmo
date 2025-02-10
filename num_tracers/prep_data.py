import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# efficiencies from this paper: https://arxiv.org/pdf/2306.06307
# LRGs (98.9%)
eff_LRG = 0.989
# assume the VLO efficiency is 95% and the LOP efficiency is 70%
eff_ELG_VLO = 0.95
eff_ELG_LOP = 0.7
# Lya QSO (65%)
eff_QSO = 0.65


desi_data = '/home/ashandonay/cobaya/packages/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
column_names = ['z', 'value_at_z', 'quantity']
data_df = pd.read_csv(desi_data, delimiter=' ', names=column_names, comment='#')
desi_df = data_df.loc[(data_df['quantity'] == 'DH_over_rs') | (data_df['quantity'] == 'DM_over_rs')]

cov_matrix = np.loadtxt('/home/ashandonay/cobaya/packages/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')
cov_matrix = cov_matrix[np.ix_(desi_df.index, desi_df.index)]
corr_matrix = cov_matrix/np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
np.save('/home/ashandonay/bed/desi_cov.npy', cov_matrix)
np.save('/home/ashandonay/bed/desi_corr.npy', corr_matrix)

desi_df.insert(0, 'tracer', '')
desi_df.insert(1, 'num', 0)
desi_df.insert(3, 'efficiency', 1)
desi_df.insert(4, 'std', 0)
# add the diagonal of the cov matrix to the dataframe
desi_df.loc[:, 'std'] = np.sqrt(np.diag(cov_matrix))

# data from table 1 of https://arxiv.org/pdf/2404.03002
desi_df.loc[desi_df['z'] == 0.51, ["tracer", "num"]] = "LRG1", 506905
desi_df.loc[desi_df['z'] == 0.706, ["tracer", "num"]] = "LRG2", 771875
desi_df.loc[desi_df['z'] == 0.930, ["tracer", "num"]] = "LRG3+ELG1", 1876164
desi_df.loc[desi_df['z'] == 1.317, ["tracer", "num"]] = "ELG2", 1415687
desi_df.loc[desi_df['z'] == 2.330, ["tracer", "num"]] = "Lya QSO", 709565

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

num_ELG2 = desi_df.loc[desi_df["tracer"] == "ELG2", "num"].iloc[0]
num_ELG1 = int(((ELG1_VLO + ELG1_LOP) / (ELG2_VLO + ELG2_LOP)) * num_ELG2)
num_LRG3ELG1 = desi_df.loc[desi_df["tracer"] == "LRG3+ELG1", "num"].iloc[0]
num_LRG3 = num_LRG3ELG1 - num_ELG1

desi_df.loc[desi_df['z'] == 0.51, "efficiency"] =  eff_LRG
desi_df.loc[desi_df['z'] == 0.706, "efficiency"] = eff_LRG
desi_df.loc[desi_df['z'] == 0.930, "efficiency"] = (num_ELG1 / num_LRG3ELG1) * ((ELG1_VLO / (ELG1_VLO + ELG1_LOP)) * eff_ELG_VLO + (ELG1_LOP / (ELG1_VLO + ELG1_LOP)) * eff_ELG_LOP) + (1 - (num_ELG1 / num_LRG3ELG1)) * eff_LRG
desi_df.loc[desi_df['z'] == 1.317, "efficiency"] = (ELG2_VLO / (ELG2_VLO + ELG2_LOP)) * eff_ELG_VLO + (ELG2_LOP / (ELG2_VLO + ELG2_LOP)) * eff_ELG_LOP
desi_df.loc[desi_df['z'] == 2.330, "efficiency"] = eff_QSO
desi_df.to_csv('/home/ashandonay/bed/desi_data.csv', index=False)

desi_tracers = pd.DataFrame({
    'tracer': ['LRG1', 'LRG2', 'LRG3', 'ELG1', 'ELG2', 'Lya QSO'],
    'class': ['LRG', 'LRG', 'LRG', 'ELG', 'ELG', 'QSO'],
    'efficiency': [eff_LRG, 
            eff_LRG, 
            eff_LRG, 
            ((ELG1_VLO / (ELG1_VLO + ELG1_LOP)) * eff_ELG_VLO + (ELG1_LOP / (ELG1_VLO + ELG1_LOP)) * eff_ELG_LOP), 
            (ELG2_VLO / (ELG2_VLO + ELG2_LOP)) * eff_ELG_VLO + (ELG2_LOP / (ELG2_VLO + ELG2_LOP)) * eff_ELG_LOP, 
            eff_QSO],
    'passed': [desi_df.loc[desi_df["tracer"] == "LRG1", "num"].iloc[0], 
            desi_df.loc[desi_df["tracer"] == "LRG2", "num"].iloc[0], 
            num_LRG3, 
            num_ELG1, 
            desi_df.loc[desi_df["tracer"] == "ELG2", "num"].iloc[0],
            desi_df.loc[desi_df["tracer"] == "Lya QSO", "num"].iloc[0]]
            })
desi_tracers['observed'] = desi_tracers['passed'] / desi_tracers['efficiency']

# normalize the passed and observed counts by the total number of tracers
desi_tracers['passed'] = desi_tracers['passed'] / desi_tracers['passed'].sum()
desi_tracers['observed'] = desi_tracers['observed'] / desi_tracers['observed'].sum()
desi_tracers.to_csv('/home/ashandonay/bed/desi_tracers.csv', index=False)