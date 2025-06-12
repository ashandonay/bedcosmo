import numpy as np
from sklearn.model_selection import train_test_split
from cosmopower import cosmopower_NN
import camb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def cosmopower_train(
        spectrum='TT', 
        epochs=1000, 
        save_path='.', 
        batch_size=100,
        lrs=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        log=False
        ):
    print(f"Training {spectrum} spectrum")
    # Load training and testing data
    if log:
        train_data = np.load(f"{save_path}/cmb_log_{spectrum}_train.npz")
        test_data = np.load(f"{save_path}/cmb_log_{spectrum}_test.npz")
    else:
        train_data = np.load(f"{save_path}/cmb_{spectrum}_train.npz")
        test_data = np.load(f"{save_path}/cmb_{spectrum}_test.npz")
    train_params = np.load(f"{save_path}/cmb_params_train.npz")
    test_params = np.load(f"{save_path}/cmb_params_test.npz")

    # Extract arrays for training and validation
    X_train = train_params
    X_test = test_params
    Y_train = train_data['features']
    Y_test = test_data['features']

    ell_range = train_data['modes']
    param_names = train_params.files
    # ['ombh2', 'omch2', 'ns', 'logA', 'tau', 'nnu', 'theta_MC_100']

    # Set up cosmopower neural network
    model = cosmopower_NN(
        parameters=param_names,
        modes=ell_range,
        n_hidden=[512, 512, 512, 512],
        verbose=True
    )
    if log:
        model_name = f'{save_path}/{spectrum}_log_cp_NN'
    else:
        model_name = f'{save_path}/{spectrum}_cp_NN'
    model.train(
        training_parameters=X_train,
        training_features=Y_train,
        filename_saved_model=model_name,
        validation_split=0.1,
        learning_rates=lrs,
        batch_sizes=len(lrs)*[batch_size],
        gradient_accumulation_steps=len(lrs)*[1],
        patience_values=len(lrs)*[100],
        max_epochs=len(lrs)*[epochs],
    )

    if log:
        predicted_testing_spectra = model.ten_to_predictions_np(X_test)
    else:
        predicted_testing_spectra = model.predictions_np(X_test)

    plt.figure()
    colors = ['tab:blue', 'tab:red', 'tab:green']
    # ['ombh2', 'omch2', 'ns', 'logA', 'tau', 'nnu', 'theta_MC_100']
    fiducial_params = {'ombh2': 0.02237,
                       'omch2': 0.12,
                       'ns': 0.9649,
                       'logA': 3.036,
                       'tau': 0.0544,
                       'nnu': 3.044,
                       'theta_MC_100': 1.04109
                       }
    plot_fiducial(fiducial_params, model, spectrum)
    for i in range(3):
        pred = predicted_testing_spectra[i]
        if log:
            true = 10.**Y_test[i]
        else:
            true = Y_test[i]
        # Plot original
        plt.semilogx(ell_range, true, color=colors[i], label=f'Original {i+1}', linewidth=2)
        # Plot NN reconstructed with a lighter shade of the same color
        lighter = lighten_color(colors[i], 0.7)
        plt.semilogx(
            ell_range, pred, 
            color=lighter, 
            label=f'NN reconstructed {i+1}', 
            linestyle='--', 
            linewidth=2
        )
    # Custom legend: one for each color, and one for the dashed style
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Fiducial Cosmology'),
        Line2D([0], [0], color=colors[0], lw=2, label='Sample 1'),
        Line2D([0], [0], color=colors[1], lw=2, label='Sample 2'),
        Line2D([0], [0], color=colors[2], lw=2, label='Sample 3'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='NN reconstructed')
    ]
    plt.xlabel('$\ell$')
    plt.ylabel('$\\frac{\ell(\ell+1)}{2 \pi} C_\ell$')
    plt.title(f'{spectrum} spectrum reconstruction')
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(f'{save_path}/reconstruction_{spectrum}.png')

def plot_fiducial(params, model, spectrum, log=False):
    
    # Unpack parameters
    ombh2 = params['ombh2']
    omch2 = params['omch2']
    ns = params['ns']
    logA = params['logA']
    tau = params['tau']
    theta_MC_100 = params['theta_MC_100']
    mnu = params.get('mnu', 0.06)
    nnu = params.get('nnu', 3.044)
    w = params.get('w', -1.0)
    wa = params.get('wa', 0.0)
    
    # Convert logA to As
    As = 1e-10 * np.exp(logA)
    cosmomc_theta = 1e-2 * theta_MC_100

    cp = camb.CAMBparams()
    cp.set_cosmology(
        ombh2=ombh2,
        omch2=omch2,
        tau=tau,
        mnu=mnu,
        nnu=nnu,
        cosmomc_theta=cosmomc_theta,
        theta_H0_range=[20, 100],
        num_massive_neutrinos=1
    )
    cp.set_dark_energy(w=w, wa=wa)
    cp.InitPower.set_params(As=As, ns=ns)
    cp.set_for_lmax(2508, lens_potential_accuracy=0)
    
    results = camb.get_results(cp)

    powers = results.get_cmb_power_spectra(cp, CMB_unit='muK', lmax=2508)
    camb_cl = powers['total']
    if spectrum == 'TT':
        cl = camb_cl[:, 0][2:]
        ell = np.arange(camb_cl.shape[0])[2:]
    elif spectrum == 'EE':
        cl = camb_cl[:, 1][2:1997]
        ell = np.arange(camb_cl.shape[0])[2:1997]
    elif spectrum == 'TE':
        cl = camb_cl[:, 3][2:1997]
        ell = np.arange(camb_cl.shape[0])[2:1997]

    if log:
        fid_pred = model.ten_to_predictions_np({k: [v] for k, v in params.items()})[0]
    else:
        fid_pred = model.predictions_np({k: [v] for k, v in params.items()})[0]
    plt.semilogx(ell, cl, color='black', label=f'CAMB', linewidth=2)
    plt.semilogx(ell, fid_pred, color='black', label=f'Fiducial', linewidth=2, linestyle='--')

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
if __name__ == "__main__":

    save_path = '/home/ashandonay/data/cosmopower'
    
    lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for s in ['TE']:
        cosmopower_train(s, epochs=2000, save_path=save_path, lrs=lrs, batch_size=100)