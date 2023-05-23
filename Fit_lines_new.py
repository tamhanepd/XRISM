import numpy as np
import lmfit
from lmfit.models import GaussianModel, LinearModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lines
import pprint
import os
import csv


class Spectrum():
    def __init__(self, specfile):
        # Initialize the Spectrum object with a specfile
        self.specfile = specfile
        
        # Load data from the specfile using numpy
        data = np.loadtxt(self.specfile, delimiter=',')
        
        # Extract wavelength, wavelength error, count rate, and count rate error from the data
        self.x = data[:, 0]     # wavelength
        self.xerr = data[:, 1]  # wavelength error
        self.y = data[:, 2]     # count rate
        self.yerr = data[:, 3]  # count rate error



class fit_line():
    def __init__(self, Spectrum):
        # Initialize the fit_line object with a Spectrum object
        self.spec = Spectrum

    def fit(self, wavelength, bound, z, title, amplitudes=None, centers=None,
            sigmas=None, ncomp=1, show=False, figname=None):
        """Fit one or more Gaussians and a linear model to the given data.

        Parameters:
            wavelength (float): The central wavelength to consider.
            bound (float): The width of the slice of the wavelength to consider.
            z (float): Redshift value.
            title (str): Title for the plot.
            amplitudes (list or array-like): Amplitude parameters for the Gaussians.
            centers (list or array-like): Center parameters for the Gaussians.
            sigmas (list or array-like): Sigma parameters for the Gaussians.
            ncomp (int, optional): Number of Gaussian components to use in the model. Default is 1.
            show (bool, optional): Whether to display the plot. Default is False.
            figname (str, optional): Name of the file to save the plot. Default is None.

        Returns:
            lmfit.Model: A model with a sum of ncomp Gaussians and a linear model.
        """
        # Adjust the wavelength based on the redshift
        wavelength = wavelength/(1+z)
        
        # Extract the data within the specified wavelength range
        idx = np.where(np.logical_and(self.spec.x >= wavelength - bound, 
                                      self.spec.x <= wavelength + bound))
        x = self.spec.x[idx]
        y = self.spec.y[idx]
        xerr = self.spec.xerr[idx]
        yerr = self.spec.yerr[idx]

        if ncomp == 1:
            # If only one Gaussian component is requested, create a GaussianModel and a LinearModel
            gauss = GaussianModel(prefix='g1_')
            linear = LinearModel()
            model = gauss + linear

            # Set initial parameter hints from the data
            amplitudes = [np.max(y)-np.min(y)]
            sigmas = [0.002]
            centers = [wavelength]
        else:
            # Define a function that creates a model with ncomp gaussians
            def composite_model(ncomp):
                model = GaussianModel(prefix='g1_')
                for i in range(ncomp - 1):
                    model += GaussianModel(prefix='g%s_'%(i+2))
                return model

            # Create the sum of ncomp Gaussians and a LinearModel
            model = composite_model(ncomp)
            model += LinearModel()

        # Set initial parameter guesses
        params = model.make_params()

        for i in range(ncomp):
            # Set initial guesses for parameters of the Gaussian components
            params['g{}_amplitude'.format(i+1)].set(value=amplitudes[i], min=0.001)
            params['g{}_center'.format(i+1)].set(value=centers[i])
            params['g{}_sigma'.format(i+1)].set(value=sigmas[i])

        # Set initial values for linear parameters
        params['slope'].set(vary=True)
        params['intercept'].set(value=np.min(y), vary=True)

        # Perform the fitting operation using lmfit
        result = model.fit(y, params, x=x, weights=1/yerr)

        # Plot the data and the fitted model
        plt.errorbar(self.spec.x[idx], self.spec.y[idx], xerr=self.spec.xerr[idx], yerr=self.spec.yerr[idx], fmt='.', color='k', alpha=0.5)
        plt.plot(self.spec.x[idx], result.best_fit, 'r-', label='fit')
        
        if ncomp > 1:
            # Plot individual components with the best-fit model
            comps = result.eval_components(x=x)
            for i in range(ncomp):
                prefix = 'g{}_'.format(i + 1)
                plt.plot(x, comps[prefix], ls='--', label='Comp {}'.format(i + 1))

        plt.xlabel('Energy (keV)')
        plt.ylabel(r'counts s$^{-1}$ keV$^{-1}$')
        plt.title(title)
        plt.legend(fontsize=12)
        
        # Save the plot to a file if specified
        if figname != None:
            plt.savefig(figname)
        
        # Display the plot if requested
        if show:
            plt.show()

        # Store the fitting result in the object's attribute
        self.result = result



    def get_fit_info(self, pout=False, ncomp=1, linename=''):
        """
        Generates a dictionary of best fit parameter values and their 1-sigma uncertainties.

        Parameters:
            pout (bool): Whether to print out the results in the terminal. Default: False
            ncomp (int): Number of Gaussian components used in the fit. Default: 6

        Returns:
            output (list): List of lists containing the fit parameter values and their 1-sigma errors for each component.
                Each inner list contains the following information:
                - Component number
                - Amplitude value
                - Positive error for amplitude
                - Negative error for amplitude
                - Center value
                - Positive error for center
                - Negative error for center
                - Sigma value
                - Positive error for sigma
                - Negative error for sigma
                - Velocity dispersion value (calculated as sigma / center * speed of light)
                - Positive error for velocity dispersion
                - Negative error for velocity dispersion
        """

        c = 299792.458 # km/s, speed of light
        table = []

        try:
            # Compute confidence intervals for the fit parameters
            ci = lmfit.conf_interval(self.result, self.result)
            
            for i in range(ncomp):
                # Extract the confidence intervals for amplitude, center, and sigma
                amp_conf_int = ci['g{}_amplitude'.format(i+1)]
                cen_conf_int = ci['g{}_center'.format(i+1)]
                sigma_conf_int = ci['g{}_sigma'.format(i+1)]
                
                # Extract the best-fit values and calculate the positive and negative errors
                amp_val = amp_conf_int[3][1]
                amp_err_pos = amp_conf_int[4][1] - amp_val
                amp_err_neg = amp_val - amp_conf_int[2][1]
                
                cen_val = cen_conf_int[3][1]
                cen_err_pos = cen_conf_int[4][1] - cen_val
                cen_err_neg = cen_val - cen_conf_int[2][1]
                
                sigma_val = sigma_conf_int[3][1]
                sigma_err_pos = sigma_conf_int[4][1] - sigma_val
                sigma_err_neg = sigma_val - sigma_conf_int[2][1]

                # Calculate velocity dispersion and its positive and negative errors
                sigma_vel_val = sigma_val / cen_val * c
                sigma_vel_err_pos = sigma_err_pos / cen_val * c
                sigma_vel_err_neg = sigma_err_neg / cen_val * c
                
                # Append the fit information for this component to the table
                table.append([linename+'_'+str(i+1), amp_val, amp_err_pos, amp_err_neg,
                              cen_val, cen_err_pos, cen_err_neg,
                              sigma_val, sigma_err_pos, sigma_err_neg,
                              sigma_vel_val, sigma_vel_err_pos, sigma_vel_err_neg])
        
        except lmfit.minimizer.MinimizerException:
            # If an exception occurs during the computation of confidence intervals,
            # use the best-fit values directly
            for i in range(ncomp):
                amp_val = self.result.params['g{}_amplitude'.format(i+1)].value
                cen_val = self.result.params['g{}_center'.format(i+1)].value
                sigma_val = self.result.params['g{}_sigma'.format(i+1)].value
                sigma_vel_val = sigma_val / cen_val * c

                # Set errors to 0 as confidence intervals could not be computed
                table.append([linename+'_'+str(i+1), amp_val, 0, 0,
                              cen_val, 0, 0,
                              sigma_val, 0, 0,
                              sigma_vel_val, 0, 0])

        if pout:
            # Print the table if pout is True
            print(*table)

        return table



    def calculate_lw(self, T, T_err, output, mass=1.2, res=5.0, ncomp=1):
        """
        Calculate the thermal, instrumental, and turbulent velocities based on the gas temperature and fitting output.

        Parameters:
            T (float): The gas temperature in keV.
            T_err (float): The error in gas temperature in keV.
            output (list): List containing the fitting parameters.
            mass (float): The gas particle mass in atomic mass units (amu). Default: 1.2 amu.
            res (float): Instrumental resolution (FWHM) in eV. Default: 5.0 eV.
            ncomp (int): Number of Gaussian components used in the fit. Default: 1.

        Returns:
            output (list): List containing the fitting parameters with appended thermal, instrumental, and turbulent velocities and errors.
            
        This function performs the following steps:
        1. Converts the gas temperature from keV to Joules.
        2. Converts the gas temperature error from keV to Joules.
        3. Calculates the gas particle mass in kilograms.
        4. Initializes arrays for wavelength, total linewidth, and total linewidth error.
        5. Extracts relevant values from the fitting output and stores them in the corresponding arrays.
        6. Calculates the thermal rms velocity in one dimension.
        7. Calculates the error in the thermal rms velocity.
        8. Calculates the instrumental linewidth.
        9. Calculates the turbulent rms velocity.
        10. Calculates the error in the turbulent rms velocity.
        11. Appends the calculated velocities and errors to the output list for each component.
        12. Returns the modified output list.
        """
        
        T = T * 1000 * 1.6e-19 # converting keV to Joules
        T_err = T_err * 1000 * 1.6e-19 # converting keV to Joules
        mass_kg = mass * 1.66e-27 # gas particle mass in kg
        c = 299792.458 # km/s, speed of light

        wavelength = np.zeros(ncomp)
        lw_tot = np.zeros(ncomp)
        lw_tot_err = np.zeros(ncomp)

        for i in range(ncomp):
            wavelength[i] = output[i][4] # keV
            lw_tot[i] = output[i][10] # km/s
            lw_tot_err[i] = abs((output[i][11]+output[i][12])/2) # km/s
        
        # Calculate the thermal rms velocity
        lw_th = np.sqrt(T / mass_kg) / 1000 # km/s in one dimension
        lw_th_err = 0.5 * np.sqrt(1 / mass_kg / T) * T_err / 1000 # km/s
        
        # Calculate instrumental linewidth
        lw_i = ((res/2.355)/(wavelength*1000) * c) # km/s

        # Calculate the turbulent rms velocity
        lw_turb = np.sqrt(lw_tot**2 - lw_th**2 - lw_i**2)
        lw_turb_err = abs(lw_tot*lw_tot_err - lw_th*lw_th_err) / lw_turb

        for i in range(ncomp):
            # Append the calculated velocities and errors to the output list
            output[i].append(lw_th)
            output[i].append(lw_th_err)
            output[i].append(lw_i[i])
            output[i].append(lw_turb[i])
            output[i].append(lw_turb_err[i])

        return output



def write_outputs_to_csv(line, result, filename):
    """
    Writes the output dictionary to a CSV file with the specified filename.
    If the file already exists, appends the new output directory values on a new line.

    Parameters:
        line (dict): A dictionary of output values.
        result (list): List of lists containing the output values for each row.
        filename (str): The name of the CSV file to write to.

    Returns:
        None
    """
    
    # Write the output values to a CSV file
    fieldnames = ['line', 'amp', 'amp_p', 'amp_n', 'center', 'center_p', 'center_n',
                  'sigma', 'sigma_p', 'sigma_n',
                  'sigma_vel', 'sigma_vel_p', 'sigma_vel_n', 'v_th', 'v_th_err',
                  'v_instr', 'v_turb', 'v_turb_err']
    
    # check if the file already exists
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(fieldnames) # write header row if file didn't exist
        for row in result:
            writer.writerow(row)



def main(specfile, spec, bound, redshift, linename, linewav, pdf_pages,
         amplitudes=None, sigmas=None, centers=None, ncomp=1, figname=None,
         T=4, T_err=0.2, mass=1.2):
    """
    Fit a single Gaussian + linear component model to a given spectrum and
    plot the input spectrum and best-fit model.

    Parameters:
        specfile (str): The input spectrum file path.
        spec (Spectrum): A Spectrum object containing the spectrum data.
        bound (float): The width of the slice of the wavelength to consider
                       for the fitting process.
        redshift (float): The redshift of the galaxy.
        linename (str): The name of the emission line.
        linewav (float): The rest-frame wavelength of the emission line.
        pdf_pages (PdfPages): A PdfPages instance where the generated plot
                              will be saved as a new page.
        amplitudes (list or None): Amplitude parameters for the Gaussians.
                                   Default: None
        sigmas (list or None): Sigma parameters for the Gaussians.
                               Default: None
        centers (list or None): Center parameters for the Gaussians.
                                Default: None
        ncomp (int): The number of Gaussian components to fit.
                     Default: 6
        figname (str or None): The name of the output figure file.
                               Default: None
        T (float): The gas temperature in keV.
                   Default: 4
        T_err (float): The error in gas temperature in keV.
                       Default: 0.2
        mass (float): The gas particle mass in atomic mass units (amu).
                      Default: 1.2

    Returns:
        None

    The function performs the following tasks:
    1. Initializes a fit_line object with the given spec parameter.
    2. Calls the fit method of the fit_line object to fit the model
       with the provided parameters.
    3. Saves the generated plot as a new page in the pdf_pages instance.
    4. Clears the current figure to avoid overlapping plots.
    5. Retrieves the fit information and prints it to the console.
    6. Writes the fit information to a CSV file with a .res extension.
       The output file has the same name as the input spectrum file,
       but with a .res extension.
    7. Calculates the velocities using the fit information and prints them.
    8. Prints a separator line to separate multiple runs of the function.
    """
    fitter = fit_line(spec)
    fitter.fit(linewav, bound, redshift, linename, amplitudes=amplitudes,
               centers=centers, sigmas=sigmas, ncomp=ncomp)

    # Save the current plot as a new page in the PDF file
    pdf_pages.savefig(plt.gcf())

    # Clear the current figure to avoid overlapping plots
    plt.clf()

    # Retrieve the fit information
    result = fitter.get_fit_info(pout=False, ncomp=ncomp, linename=linename)
    print(linename)
    print("\n")

    # Calculate velocities using the fit information
    output = fitter.calculate_lw(T, T_err, result, mass=mass, ncomp=ncomp)

    # Write the fit information to a CSV file
    write_outputs_to_csv(linename, output, specfile[:-4]+'.res')

    # Print a separator line
    print("--------------------------------------------\n")


#---------------------------------------------------------------

# root = 'a1795'
specfile = '/image_nh0p041_v100_exp1000_Z0p5.csv'

# Create the spec object
spec = Spectrum(specfile)

# Delete the existing CSV file with output of spectral fits before
# creating a new one
if os.path.isfile(specfile[:-4]+'.res'):
    os.remove(specfile[:-4]+'.res')

output_pdf = specfile[:-4]+'_LinePlots.pdf'

# Create a PdfPages instance
pdf_pages = PdfPages(output_pdf)


#---------------------------------------------------------------

# Fit OVIII line
main(specfile, spec, 0.07, 0.063001, 'OVIII', lines.O_VIII, pdf_pages, mass=15.999)

#---------------------------------------------------------------

# Ne X line:

main(specfile, spec, 0.02, 0.063001, 'Ne X', lines.Ne_X, pdf_pages, mass=20.1797)

#---------------------------------------------------------------

# Mg XII line:

main(specfile, spec, 0.02, 0.063001, 'Mg XII', lines.Mg_XII, pdf_pages, mass=24.305)

#---------------------------------------------------------------

# Si XIV line:

main(specfile, spec, 0.02, 0.063001, 'Si XIV', lines.Si_XIV, pdf_pages, mass=28.0855)

#---------------------------------------------------------------

# Fe lines around 1 keV:

amplitudes=[3, 6, 3, 13, 8, 13]
centers = [0.992, 1.02, 1.035, 1.038, 1.058, 1.097]
sigmas = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]

main(specfile, spec, 0.08, 0.063001, 'Fe_XXIV', lines.Fe_XXIV_b, pdf_pages,
    amplitudes=amplitudes, centers=centers, sigmas=sigmas, ncomp=6,
    mass=55.847)

#---------------------------------------------------------------

# Fe K line and other lines around it:

amplitudes=[0.5,3.5,1.5,2.5,2,10]
centers = [6.227, 6.247, 6.265, 6.275, 6.289, 6.307]
sigmas = [0.005, 0.01, 0.005, 0.005, 0.005, 0.008]

main(specfile, spec, 0.08, 0.063001, 'Fe_K', lines.Fe_K, pdf_pages,
    amplitudes=amplitudes, centers=centers, sigmas=sigmas, ncomp=6,
    mass=55.847)


# Close the PDF file
pdf_pages.close()
