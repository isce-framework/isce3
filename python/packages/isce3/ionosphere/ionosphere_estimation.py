import os
import journal
import numpy as np
from osgeo import gdal
import copy
from scipy.signal import resample
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, distance_transform_edt

from nisar.workflows.filter_data import filter_data
from nisar.workflows.filter_interferogram import create_gaussian_kernel


class IonosphereEstimation:
    '''
    Esimate ionospheric phase screen
    '''
    def __init__(self, 
                 main_center_freq=None, 
                 side_center_freq=None, 
                 low_center_freq=None, 
                 high_center_freq=None,
                 method=None): 
                 
        """Initialized IonosphererEstimation Class
        
        Parameters
        ----------
        main_center_freq : float
            center frequency of main band (freqA) [Hz]
        side_center_freq : float
            center frequency of side band (freqB) [Hz]
        low_center_freq : float 
            center frequency of lower sub-band [Hz]
        high_center_freq : float
            center frequency of upper sub-band [Hz]
        method : {'split_main_band', 'main_side_band', 
            'main_diff_ms_band'}
            ionosphere estimation method
        """       

        error_channel = journal.error('ionosphere.IonosphereEstimation')

        if method not in ['split_main_band', 'main_side_band', 
            'main_diff_ms_band']:
            err_str = f"{method} not a valid diversity method type"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.diversity_method = method

        # Center frequency for frequency A is needed for all methods. 
        if main_center_freq is None:
            err_str = f"Center frequency for frequency A "\
                f" is needed for {method}"
            error_channel.log(err_str)
            raise ValueError(err_str)
        
        # Center frequency for frequency B is needed except 
        # split_main_band. 
        if side_center_freq is None:
            if method in ['main_side_band', 'main_diff_band']:
                err_str = f"Center frequency for frequency B"\
                f" is needed for {method}"
                error_channel.log(err_str)
                raise ValueError(err_str)

        # Center frequencies for sub-bands are needed except 
        # main_side_band method.
        if (low_center_freq is None) or (high_center_freq is None):
            if method in ['split_main_band', 'main_diff_band']:
                err_str = f"Center frequency for frequency A"\
                    f" is needed for {method}"
                error_channel.log(err_str)
                raise ValueError(err_str)
            
        self.f0 = main_center_freq
        self.f1 = side_center_freq
        self.freq_low = low_center_freq
        self.freq_high = high_center_freq

    def compute_disp_nondisp(self, 
            phi_sub_low=None, 
            phi_sub_high=None,
            phi_main=None, 
            phi_side=None,
            slant_main=None,
            slant_side=None, 
            comm_unwcor_coef=None,
            diff_unwcor_coef=None, 
            no_data=0):

        """Estimates dispersive and non-dispersive phase using given  
        spectral diversity method. Note that each methods require different
        unwrapped interferograms.
        - split_main_band requires [phi_sub_low, phi_sub_high]
        - main_side_band requires [phi_main, phi_side]
        - main_diff_ms_band requires [phi_main, phi_side]

        If correction coefficients are given, phase is corrected. 

        Parameters
        ----------
        phi_sub_low : numpy.ndarray 
            unwrapped phase array of low sub-band interferogram  
        phi_sub_high : numpy.ndarray 
            unwrapped phase array of high sub-band interferogram 
        phi_main : numpy.ndarray  
            unwrapped phase array of frequency A interferogram  
        phi_side : numpy.ndarray  
            unwrapped phase array of frequency B interferogram 
        slant_main : numpy.ndarray  
            slant range array of frequency A interferogram 
        slant_side : numpy.ndarray  
            slant range array of frequency B interferogram             
        comm_unwcor_coef : numpy.ndarray  
            common correction coefficient of unwrapped phases
        diff_unwcor_coef : numpy.ndarray  
            differential correction coefficient of unwrapped phases
        no_data : float
            no data value
        
        Returns
        -------
        dispersive : numpy.ndarray 
            numpy array of dispersive array
        non_dispersive : numpy.ndarray 
            non-dispersive phase array
        """
        
        if self.diversity_method == 'split_main_band':
            if (phi_sub_low is None) or (phi_sub_high is None):
                err_str = f"sub-band unwrapped interferogram array is required."
                error_channel.log(err_str)
                raise ValueError(err_str)
            phi_low = phi_sub_low
            phi_high = phi_sub_high
            # set up mask for areas where no-data values are located
            no_data_array = (phi_sub_high==no_data) |\
                            (phi_sub_low==no_data)

        if self.diversity_method in ['main_side_band', 'main_diff_ms_band']:
            # When side-band arrays is used,
            # arrays should be resampled to have the same size with side-band arrays
            if (phi_main is None) or (phi_side is None):
                err_str = f"unwrapped interferogram array main and side band"\
                    "is required."
                error_channel.log(err_str)
                raise ValueError(err_str)
            
            else:
                phi_main = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    phi_main)
                self.slant_main = slant_main
                self.slant_side = slant_side

                if phi_sub_low is not None:
                    phi_sub_low = self.resample_freqA_array(
                        slant_main, 
                        slant_side, 
                        phi_sub_low)
                if phi_sub_high is not None:
                    phi_sub_high = self.resample_freqA_array(
                        slant_main, 
                        slant_side, 
                        phi_sub_high)  

                phi_low = phi_main
                phi_high = phi_side
                no_data_array = (phi_main==no_data) |\
                                (phi_side==no_data)


        # correct unwrapped phase when correction coefficients are given
        if (comm_unwcor_coef is not None) and \
           (diff_unwcor_coef is not None):

            phi_low = phi_low - 2 * np.pi * comm_unwcor_coef
            phi_high = phi_high - 2 * np.pi *\
                (comm_unwcor_coef + diff_unwcor_coef)
                         
        if self.diversity_method == 'split_main_band':
            dispersive, non_dispersive = self.estimate_iono_low_high(
                f0=self.f0, 
                freq_low=self.freq_low, 
                freq_high=self.freq_high, 
                phi0_low=phi_low, 
                phi0_high=phi_high)

        if self.diversity_method == 'main_side_band':
            dispersive, non_dispersive = self.estimate_iono_main_side(
                f0=self.f0, 
                f1=self.f1, 
                phi0=phi_low, 
                phi1=phi_high)

        if self.diversity_method == 'main_diff_ms_band':
            dispersive, non_dispersive = self.estimate_iono_main_diff(
                f0=self.f0, 
                f1=self.f1, 
                phi0=phi_low, 
                phi1=phi_high)

        dispersive[no_data_array] = no_data
        non_dispersive[no_data_array] = no_data

        return dispersive, non_dispersive

    def estimate_iono_low_high(self, 
            f0, 
            freq_low, 
            freq_high, 
            phi0_low, 
            phi0_high):

        """Estimates ionospheric phase from low and high sub-band 
        interferograms i.e. split_main_band method
        
        Parameters
        ----------
        f0 : float
            radar center frequency of frequency A band
        freq_low : float 
            radar center frequency of lower sub-band
        freq_high : float
            radar center frequency of upper sub-band
        phi0_low : numpy.ndarray 
            numpy array of lower sub-band interferogram
        phi0_high : numpy.ndarray 
            numpy array of upper sub-band interferogram
        
        Returns
        -------
        dispersive : numpy.ndarray 
            numpy array of estimated dispersive 
        non_dispersive : numpy.ndarray 
            numpy array of estimated non-dispersive 
        """       
     
        y_size, x_size = phi0_low.shape
        d = np.ones((2, y_size * x_size))
        d[0,:] = phi0_low.flatten()
        d[1,:] = phi0_high.flatten()
        coeff_mat = np.ones((2, 2))

        coeff_mat[0, 0] = freq_low / f0
        coeff_mat[0, 1] = f0 / freq_low
        coeff_mat[1, 0] = freq_high / f0
        coeff_mat[1, 1] = f0 / freq_high
        coeff_mat1 = np.linalg.pinv(coeff_mat)
        output = np.dot(coeff_mat1, d)
        
        non_dispersive = output[0, :].reshape(y_size, x_size)
        dispersive = output[1].reshape(y_size, x_size)

        return dispersive, non_dispersive

    def estimate_iono_main_side(self, 
            f0, 
            f1, 
            phi0, 
            phi1):
 
        """Estimates ionospheric phase from frequency A and B 
        interferograms
        
        Parameters
        ----------
        f0 : float
            radar center frequency of frequency A band
        f1 : float 
            radar center frequency of frequency B band
        phi0 : numpy.ndarray 
            numpy array of frequency A interferogram
        phi1 : numpy.ndarray 
            numpy array of frequency B interferogram
        
        Returns
        -------
        dispersive : numpy.ndarray 
            numpy array of estimated dispersive 
        non_dispersive : numpy.ndarray 
            numpy array of estimated non-dispersive 
        """       
     
        y_size, x_size = phi0.shape
        d = np.ones((2, y_size * x_size))
        d[0, :] = phi0.flatten()
        d[1, :] = phi1.flatten()
        
        coeff_mat = np.ones((2, 2))
        coeff_mat[1, 0] = f1 / f0
        coeff_mat[1, 1] = f0 / f1
        
        coeff_mat1 = np.linalg.pinv(coeff_mat)
        output = np.dot(coeff_mat1, d)
        Covx = np.linalg.inv(np.dot(coeff_mat.T, coeff_mat))

        non_dispersive = output[0].reshape(y_size, x_size)
        dispersive = output[1].reshape(y_size, x_size)

        return dispersive, non_dispersive

    def estimate_iono_main_diff(self, 
                                f0, 
                                f1, 
                                phi0, 
                                phi1):
        
        """Estimates ionospheric phase from main-band 
        and the difference of main and side-band interferograms
        
        Parameters
        ----------
        f0 : float
            radar center frequency of frequency A band
        f1 : float 
            radar center frequency of frequency B band
        phi0 : numpy.ndarray 
            numpy array of frequency A interferogram
        phi1 : numpy.ndarray 
            numpy array of frequency B interferogram
        
        Returns
        -------
        dispersive : numpy.ndarray 
            numpy array of estimated dispersive 
        non_dispersive : numpy.ndarray 
            numpy array of estimated non-dispersive 
        """     

        phi_diff = phi0 - phi1
        y_size, x_size = phi0.shape
        d = np.ones((2, y_size * x_size))
        d[0, :] = phi0.flatten()
        d[1, :] = phi_diff.flatten()
        
        coeff_mat = np.ones((2, 2))
        coeff_mat[1,0] = (f1 - f0) / f1
        coeff_mat[1,1] = (f0 - f1) / f0
        coeff_mat1 = np.linalg.pinv(coeff_mat)
        output = np.dot(coeff_mat1, d)
        cov_x = np.linalg.inv(np.dot(coeff_mat.T, coeff_mat))
        
        non_dispersive = output[1].reshape(y_size, x_size)
        dispersive = output[0].reshape(y_size, x_size)
       
        return dispersive, non_dispersive
        
    def resample_freqA_array(self, 
                            slant_main, 
                            slant_side, 
                            target_runw):
        """Resample frequency A RUNW
        
        Parameters
        ----------
        slant_main : numpy.ndarray 
            slant range array of frequency A band
        slant_side : numpy.ndarray 
            slant range array of frequency B band
        target_runw : numpy.ndarray 
            RUNW array of frequency A band
                
        Returns
        -------
        resampled_array : numpy.ndarray 
            resampled RUNW array
        """     
        height, width = target_runw.shape
        
        first_index = np.argmin(np.abs(slant_main - slant_side[0]))
        spacing_main = slant_main[1] - slant_main[0]
        spacing_side = slant_side[1] - slant_side[0]

        resampling_scale_factor = int(np.round(spacing_side / spacing_main))
        
        sub_width = int(width / resampling_scale_factor)
        x_cand = np.arange(1, width + 1)
        
        # find the maximum of the multiple of resampling_scale_factor
        resample_width_end = np.max(x_cand[x_cand % resampling_scale_factor == 0])
        resampled_array = target_runw[
            :, first_index:resample_width_end:resampling_scale_factor]
        
        return resampled_array
     
    def get_mask_array(self, 
            main_array=None,
            side_array=None, 
            low_band_array=None, 
            high_band_array=None,
            slant_main=None,
            slant_side=None,  
            threshold=0.5,
            mask_type='coherence'):
        """Get mask from coherence or connected components
        
        Parameters
        ----------
        main_array : numpy.ndarray 
            coherence of main-band interferogram
        side_array : numpy.ndarray 
            coherence of side-band interferogram
        low_band_array : numpy.ndarray 
            coherencen of main-band interferogram
        high_band_array : numpy.ndarray 
            coherence of side-band interferogram
        slant_main : numpy.ndarray 
            slant range array of frequency A band
        slant_side : numpy.ndarray 
            slant range array of frequency B band
        threshold : float
            thresholds for coherence
        mask_type : str {'coherence', 'connected_components}
            input data to be used for mask construction
        Returns
        -------
        mask_array : numpy.ndarray 
            2D mask array extracted from coherence or 
            connected components
        """     
        # Resample coherence or connected components 
        # when side array is also used. 
        if side_array is not None:
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side
            
            main_array = self.resample_freqA_array(
                slant_main, 
                slant_side, 
                main_array)
            if low_band_array is not None:
                low_band_array = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    low_band_array)
            if high_band_array is not None:
                high_band_array = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    high_band_array)
 
        if mask_type == 'connected_components':
            threshold = 0

        if self.diversity_method == 'split_main_band':
            mask_array = (high_band_array > threshold) & \
                         (low_band_array > threshold)

        elif self.diversity_method in ['main_side_band',
            'main_diff_ms_band']:
            mask_array = (main_array > threshold) & \
                         (side_array > threshold)
                
        return mask_array

    def get_mask_median_filter(self, 
            disp,
            looks,
            threshold):
        """Get mask using median filter
        
        Parameters
        ----------
        disp : numpy.ndarray 
            2D dispersive array    
        looks : int
            number of looks
        threshold : float
            coherence threshold to be used for std calculation
        
        Returns
        -------
        mask_array : numpy.ndarray 
            2D mask array extracted from coherence or 
            connected components
        """     

        std_iono, _ = self.estimate_iono_std(
            main_coh=threshold, 
            side_coh=threshold,
            low_band_coh=threshold, 
            high_band_coh=threshold,  
            number_looks=looks, 
            resample_flag=False)
            
        median_filter_size = 15
        mask_array = np.abs(disp - median_filter(disp, 
            median_filter_size)) < 3 * std_iono 

        return mask_array

    def estimate_iono_std(
            self, 
            main_coh=None, 
            side_coh=None,
            low_band_coh=None, 
            high_band_coh=None,  
            slant_main=None,
            slant_side=None, 
            number_looks=1, 
            resample_flag=True):
        """Calculate the theoretical standard deviation of 
        the ionospheric phase based on the coherencess

        Parameters
        ----------
        main_coh : numpy.ndarray 
            coherence of main-band interferogram
        side_coh : numpy.ndarray 
            coherence of side-band interferogram
        low_band_coh : numpy.ndarray 
            coherencen of main-band interferogram
        high_band_coh : numpy.ndarray 
            coherence of side-band interferogram
        slant_main : numpy.ndarray 
            slant range array of frequency A band
        slant_side : numpy.ndarray 
            slant range array of frequency B band
        number_looks : int
            number of looks
        resample_flag : bool
                            
        Returns
        -------
        sig_phi_iono : numpy.ndarray 
            phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray 
            phase standard deviation of non-dispersive
        """       
        # resample coherences array of frequency A to 
        # frequency B grid
        if (side_coh is not None) and (resample_flag):
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side
        
            main_coh = self.resample_freqA_array(
                slant_main, 
                slant_side, 
                main_coh)
            if low_band_coh is not None:
                low_band_coh = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    low_band_coh)
            if high_band_coh is not None:
                high_band_coh = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    high_band_coh)
                        
        # estimate sigma from sub-band coherences
        if (low_band_coh is not None) & (high_band_coh is not None):      
            sig_phi_low = np.sqrt(1 - low_band_coh**2) / \
                low_band_coh / np.sqrt(2 * number_looks)
            sig_phi_high = np.sqrt(1 - high_band_coh**2) / \
                high_band_coh / np.sqrt(2 * number_looks)

        # estimate sigma from main- and side- band coherences
        if (main_coh is not None) & (side_coh is not None):      
            
            sig_phi_main = np.divide(np.sqrt(1 - main_coh**2),
                main_coh / np.sqrt(2 * number_looks),
                out=np.zeros_like(main_coh),
                where=main_coh!=0)
            sig_phi_side = np.divide(np.sqrt(1 - side_coh**2),
                side_coh / np.sqrt(2 * number_looks),
                out=np.zeros_like(side_coh),
                where=side_coh!=0)

   
        if self.diversity_method == 'split_main_band':
            sig_phi_iono, sig_nondisp = \
                self.estimate_sigma_split_main_band(
                sig_phi_low, 
                sig_phi_high)

        elif self.diversity_method == 'main_side_band':
            sig_phi_iono, sig_nondisp = \
                self.estimate_sigma_main_side(
                sig_phi_main, 
                sig_phi_side)
        
        elif self.diversity_method == 'main_diff_ms_band':
            sig_phi_iono, sig_nondisp = \
                self.estimate_sigma_main_diff(
                sig_phi_main, 
                sig_phi_side)

        return sig_phi_iono, sig_nondisp

    def estimate_sigma_split_main_band(
            self, 
            sig_phi_low, 
            sig_phi_high):
        """Estimate sigma from coherence for split_main_band method
        
        Parameters
        ----------
        sig_phi_low : numpy.ndarray 
            phase standard deviation of low sub-band interferogram
        sig_phi_high : numpy.ndarray 
            phase standard deviation of high sub-band interferogram
                
        Returns
        -------
        sig_iono : numpy.ndarray 
            2D phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray
            2D array of phase standard deviation of non-dispersive 
        """     
        coeff = self.freq_low * self.freq_high / self.f0 /\
            (self.freq_high**2 - self.freq_low**2)
        sig_iono = np.sqrt(coeff**2 * (self.freq_high**2 * sig_phi_low**2 
            + self.freq_low**2 * sig_phi_high**2))
        
        coef_non = self.f0 / (self.freq_high**2 - self.freq_low**2)

        sig_nondisp = np.sqrt((coef_non**2) * (self.freq_low**2) *\
                (sig_phi_low**2) + (coef_non**2) *\
                (self.freq_high**2) * (sig_phi_high**2))
        
        return sig_iono, sig_nondisp

    def estimate_sigma_main_side(
            self, 
            sig_phi0, 
            sig_phi1):
        """Estimate sigma from coherence for main-side method
        
        Parameters
        ----------
        sig_phi0 : numpy.ndarray 
            phase standard deviation of main-band interferogram
        sig_phi1 : numpy.ndarray 
            phase standard deviation of side-band interferogram
                
        Returns
        -------
        sig_iono : numpy.ndarray 
            phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray
            2D array of phase standard deviation of non-dispersive 
        """     
        a = (self.f1**2) / (self.f1**2 - self.f0**2)
        b = (self.f0 * self.f1) / (self.f1**2 - self.f0**2)
        c = (self.f0) / (self.f1**2 - self.f0**2)

        sig_iono = np.sqrt(a**2 * sig_phi0**2 + b**2 * sig_phi1**2)
        sig_non_disp = np.sqrt(c**2 * (self.f0**2 * sig_phi0**2 \
            + self.f1**2 * sig_phi1**2))
        
        return sig_iono, sig_non_disp

    def estimate_sigma_main_diff(
            self, 
            sig_phi0, 
            sig_phi1):
        """Estimate sigma from coherence for main_diff_ms method
        
        Parameters
        ----------
        sig_phi0 : numpy.ndarray 
            phase standard deviation of main-band interferogram
        sig_phi1 : numpy.ndarray 
            phase standard deviation of side-band interferogram
                
        Returns
        -------
        sig_iono : numpy.ndarray 
            2D array of phase standard deviation of dispersive 
        sig_nondisp : numpy.ndarray
            2D array of phase standard deviation of non-dispersive 
        """     

        a = self.f1 / (self.f1 + self.f0)
        b = (self.f0 * self.f1) / (self.f0**2 - self.f1**2)
        sig_phi01 = np.sqrt(sig_phi0**2 + sig_phi1**2)
        sig_iono = np.sqrt(a**2 * sig_phi0**2 + b**2 * sig_phi01**2)

        c = self.f0**2 /(self.f0**2 - self.f1**2)
        d = self.f0 * self.f1 / (self.f0**2 - self.f1**2)
        sig_nondisp = np.sqrt( c**2 * sig_phi0**2 + d**2 * sig_phi01**2)

        return sig_iono, sig_nondisp

    def compute_unwrapp_error(
            self, 
            disp_array, 
            nondisp_array,
            main_runw=None, 
            side_runw=None,
            low_sub_runw=None, 
            high_sub_runw=None, 
            y_ref=None, 
            x_ref=None):
        """Compute unwrapping error coefficients 

        Parameters
        ----------
        disp_array : numpy.ndarray
            2D dispersive array estimated from given methods
        nondisp_array : numpy.ndarray
            2D non-dispersive array estimated from given methods
        main_runw : numpy.ndarray 
            2D runw array of main-band interferogram
        side_runw : numpy.ndarray 
            2D runw array of of side-band interferogram
        low_sub_runw : numpy.ndarray 
            2D runw array of low sub-band interferogram
        high_sub_runw : numpy.ndarray 
            2D runw array of high sub-band interferogram
                            
        Returns
        -------
        com_unw_coeff : numpy.ndarray 
            2D common unwrapping error coefficient array 
        diff_unw_coeff : numpy.ndarray 
            2D differential unwrapping error coefficient array 
        """     
        # resample coherences array of frequency A to 
        # frequency B grid    
        if side_runw is not None:
            main_runw = self.resample_freqA_array(
                self.slant_main, 
                self.slant_side, 
                main_runw)

            if low_sub_runw is not None:
                low_sub_runw = self.resample_freqA_array(
                    self.slant_main, 
                    self.slant_side, 
                    low_sub_runw)

            if high_sub_runw is not None:
                high_sub_runw = self.resample_freqA_array(
                    slant_main, 
                    slant_side, 
                    high_sub_runw)

        if self.diversity_method == 'split_main_band':
            freq_diff = self.freq_high - self.freq_low
            freq_multi = self.freq_high * self.freq_low
            
            diff_unw_coeff = np.round(((high_sub_runw) - (low_sub_runw)\
                - (freq_diff / self.f0) * nondisp_array \
                + ( self.f0 * freq_diff / freq_multi) * disp_array) /\
                 2.0 / np.pi)

            com_unw_coeff = np.round((low_sub_runw + high_sub_runw \
                - 2.0 * nondisp_array - 2.0 * disp_array ) / 4.0 / np.pi\
                - diff_unw_coeff / 2)
        
        elif self.diversity_method == 'main_side_band':
            diff_unw_coeff = np.round( ( (1 - self.f1 / self.f0) \
                * nondisp_array + (1 - self.f0 / self.f1) * disp_array
                + side_runw - main_runw) / (2 * np.pi))
            com_unw_coeff = np.round( ( main_runw + side_runw \
                - (1 + self.f1 / self.f0) * nondisp_array \
                - (1 + self.f0 / self.f1) * disp_array \
                - 2 * np.pi * diff_unw_coeff) / (4 * np.pi) )

        elif self.diversity_method == 'main_diff_ms_band':
            diff_unw_coeff = np.round( ( (1 - self.f1 / self.f0) \
                * nondisp_array + (1 - self.f0 / self.f1) * disp_array
                + side_runw - main_runw) / (2 * np.pi))
            com_unw_coeff = np.round( (main_runw - nondisp_array \
                - disp_array) / (2 * np.pi) )

        return com_unw_coeff, diff_unw_coeff

class IonosphereFilter:
    '''
    Filter ionospheric phase screen
    '''
    def __init__(self, 
                 x_kernel, 
                 y_kernel, 
                 sig_x, 
                 sig_y, 
                 iteration=0,
                 filling_method='nearest', 
                 outputdir='.'):
        """Initialized IonosphereFilter with filter options

        Parameters
        ----------
        x_kernel : int
            x kernel size for gaussian filtering 
        y_kernel : int
            y kernel size for gaussian filtering 
        sig_x : int
            x standard deviation for gaussian window
        sig_y : int
            y standard deviation for gaussian window
        iteration : int
            number of iterations for filtering
        filling_method : str {'nearest', 'smoothed'}
            filling gap method for masked area
        outputdir : str
            output directory for filtered dispersive
        """
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.sig_x = sig_x
        self.sig_y = sig_y
        self.iteration = iteration
        self.filling_method = filling_method
        self.outputdir = outputdir
        
    def low_pass_filter(self, 
            input_array, 
            input_sig, 
            mask):
        """Apply low_pass_filtering for dispersive and nondispersive
        with standard deviation. Before filtering, fill the gaps with 
        smoothed or nearest values.

        Parameters
        ----------
        input_array : numpy.ndarray
            2D dispersive or nondispersive array
        input_sig : numpy.ndarray
            2D standard deviation array of dispersive 
            or nondispersive array
        mask : numpy.ndarray
            2D mask image 
        
        Returns 
        -------
        filt_data : numpy.ndarray
            2D filtered image
        filt_data_sig : numpy.ndarray
            2D filtered standard deviation image
        """
        data = input_array
        data_sig = input_sig
        data[mask==0] = np.nan
        data_sig[mask==0] = np.nan

        # filling gaps with smoothed or nearest values
        if self.filling_method == "smoothed":
            filled_data = self.fill_with_smoothed(data)
            filled_data_sig = self.fill_with_smoothed(data_sig) 

        elif self.filling_method == "nearest":
            filled_data = self.fill_nearest(data)
            filled_data_sig = self.fill_nearest(data_sig)
        
        # if self.filling_method == "None":
        filt_data, filt_data_sig = self.filter_data_with_sig(
            input_array=filled_data, 
            sig_array=filled_data_sig, 
            kernel_width=self.x_kernel, 
            kernel_length=self.y_kernel, 
            sig_kernel_x=self.sig_x, 
            sig_kernel_y=self.sig_y, 
            tempdir=self.outputdir)

        for iter_cnt in range(self.iteration):
            filt_data[mask==0] = np.nan
            
            if self.filling_method == "smoothed":
                filt_data = self.fill_with_smoothed(filt_data)

            elif self.filling_method == "nearest":
                filt_data = self.fill_nearest(filt_data)

            # Replace the valid pixels with original unfiltered data
            # to avoid too much smoothed signal
            filt_data[mask==1] = data[mask==1]
            filt_data, filt_data_sig = self.filter_data_with_sig(
                input_array=filt_data, 
                sig_array=filt_data_sig, 
                kernel_width=self.x_kernel, 
                kernel_length=self.y_kernel, 
                sig_kernel_x=self.sig_x, 
                sig_kernel_y=self.sig_y, 
                tempdir=self.outputdir)
            
        return filt_data, filt_data_sig

    def filter_data_with_sig(
            self, 
            input_array, 
            sig_array, 
            kernel_width, 
            kernel_length, 
            sig_kernel_x, 
            sig_kernel_y, 
            tempdir):
        """ 
        Parameters
        ----------
        input_array : numpy.ndarray
            2D dispersive or nondispersive array
        sig_array : numpy.ndarray
            2D standard deviation array of dispersive 
            or nondispersive array
        kernel_width : int
            x kernel size for gaussian filtering 
        kernel_length : int
            y kernel size for gaussian filtering 
        sig_kernel_x : int
            x standard deviation for gaussian window
        sig_kernel_y : int
            y standard deviation for gaussian window
        tempdir : str
            output directory for filtered dispersive
        
        Returns 
        -------
        filt_data : numpy.ndarray
            2D filtered image
        filt_data_sig : numpy.ndarray
            2D filtered standard deviation image
        """
        kernel_rows = create_gaussian_kernel(kernel_length, sig_kernel_y)
        kernel_rows = np.reshape(kernel_rows, (len(kernel_rows), 1))
        kernel_cols = create_gaussian_kernel(kernel_width, sig_kernel_x)
        kernel_cols = np.reshape(kernel_cols, (1, len(kernel_cols)))

        temp_array = os.path.join(tempdir, 'data')
        temp_filt_array = os.path.join(tempdir, 'data.filt')

        array_rows, array_cols = input_array.shape
        sig_array_sqr = sig_array**2
        input_div_sig = np.divide(input_array,
            sig_array_sqr,
            out=np.zeros_like(input_array),
            where=sig_array_sqr!=0)

        write_array(temp_array, input_div_sig, data_type=gdal.GDT_Float32)
        
        # filter_data requires the input path and output path string. 
        filter_data(temp_array, lines_per_block=200,
            kernel_rows=kernel_rows, 
            kernel_cols=kernel_cols, 
            output_data=temp_filt_array, mask_path=None)

        w1_array = os.path.join(tempdir, 'weight1')
        w1_filt_array = os.path.join(tempdir, 'weight1.filt')
        w2_filt_array = os.path.join(tempdir, 'weight2.filt')

        inv_sig = np.divide(1,
            sig_array_sqr,
            out=np.zeros_like(sig_array_sqr),
            where=sig_array_sqr!=0)
        write_array(w1_array, inv_sig, data_type=gdal.GDT_Float32)

        filter_data(w1_array, lines_per_block=200,
            kernel_rows=kernel_rows, 
            kernel_cols=kernel_cols, 
            output_data=w1_filt_array, mask_path=None)

        filter_data(w1_array, lines_per_block=200,
            kernel_rows=kernel_rows**2, 
            kernel_cols=kernel_cols**2, 
            output_data=w2_filt_array, mask_path=None)

        ds = gdal.Open(temp_filt_array, gdal.GA_ReadOnly)
        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        ds = gdal.Open(w1_filt_array, gdal.GA_ReadOnly)
        w1 = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        ds = gdal.Open(w2_filt_array, gdal.GA_ReadOnly)
        w2 = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        
        result1 = np.divide(data, w1,
            out=np.zeros_like(data),
            where=w1!=0)

        result2 = np.divide(w2, w1**2,
            out=np.zeros_like(w2),
            where=w1!=0)
        result2 = np.sqrt(result2)

        output_rows, output_cols = result1.shape

        if (output_rows != array_rows) or (output_rows != array_rows):
            result1 = result1[:array_rows, :array_cols]
            result2 = result2[:array_rows, :array_cols]
        filt_data = result1
        filt_data_sig = result2

        os.remove(w1_array)
        os.remove(w1_filt_array)
        os.remove(w2_filt_array)
        os.remove(temp_filt_array)
        os.remove(temp_array)

        return filt_data, filt_data_sig
 
    def fill_nearest(self, data, invalid=None):
        """Replace the value of invalid 'data' cells (indicated by 'invalid')
        by the value of the nearest valid data cell
        
        Parameters
        ----------
        data : numpy.ndarray
            array containing holes to be filled. 
        invalid: 
            a binary array of same shape as 'data'.
            data value are replaced where invalid is True
            If None (default), use: invalid  = np.isnan(data)
    
        Returns 
        -------   
        output: numpy.ndarray
            filled array.
        """
        if invalid is None: 
            invalid = np.isnan(data)

        ind = distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
        output = data[tuple(ind)]
        return output

    def fill_with_smoothed(self, data):
        """Replace the value of nan 'data' cells 
        by the value of the interpolated data cell
        
        Parameters
        ----------
        data : numpy.ndarray
            array containing holes to be filled. 
            nan values are considered as holes. 

        Returns 
        -------   
        data_filt2 : numpy.ndarray
            filled array.
        """
        rows, cols = data.shape
        x = np.arange(0, cols)
        y = np.arange(0, rows)
        xx, yy = np.meshgrid(x, y)

        xx = xx.ravel()              
        yy = yy.ravel()
        data = data.ravel()
        # find x and y where valid values are located. 
        xx_wo_nan = list(xx[np.invert(np.isnan(data))])  
        yy_wo_nan = list(yy[np.invert(np.isnan(data))])
        data_wo_nan = list(data[np.invert(np.isnan(data))])

        xnew = list(xx[np.isnan(data)])   
        ynew = list(yy[np.isnan(data)])

        if xnew:
            # linear interpolation with griddata
            znew = griddata((xx_wo_nan, yy_wo_nan), 
                            data_wo_nan, 
                            (xnew, ynew), 
                            method='linear')

            data_filt = data.copy()
            data_filt[np.isnan(data)] = znew
            cnt2 = np.sum(np.count_nonzero(np.isnan(data_filt)))
            loop = 0

            while (cnt2!=0 & loop<100):
                loop += 1
                idx2= np.isnan(data_filt)
            
                xx_wo_nan = list(xx[np.invert(np.isnan(data_filt))])  
                yy_wo_nan = list(yy[np.invert(np.isnan(data_filt))])
                data_wo_nan = list(data_filt[np.invert(np.isnan(data_filt))])
                xnew = list(xx[np.isnan(data_filt)])   
                ynew = list(yy[np.isnan(data_filt)])

                # nearest interpolation for extrapolation
                znew2 = griddata((xx_wo_nan, yy_wo_nan), data_wo_nan, (xnew, ynew), method='nearest')
                data_filt[np.isnan(data_filt)] = znew2
                cnt2 = np.sum(np.count_nonzero(np.isnan(data_filt)))

            data_filt2 = data_filt.reshape([rows, cols])
        else:
            data_filt2 = data

        return data_filt2


def write_array(output_str, 
        input_array, 
        data_type, 
        data_shape=None, 
        block_row=0):
    """write block array to file with gdal
    
    Parameters
    ----------
    output_str : str 
        output file name with path 
    input_array : numpy.ndarray
        2D array to be written to file 
    data_type : str 
        gdal raster type
    data_shape : list 
        raster shape, [rows, cols]
    block_row : int
        block index
    """
    if data_shape is None:
        rows, cols = input_array.shape
    else:
        rows, cols = data_shape

    if block_row == 0:
        if os.path.exists(output_str):
            os.remove(output_str)
        driver = gdal.GetDriverByName('GTiff')
        ds_data = driver.Create(output_str, cols, rows, 1, data_type)
        ds_data.WriteArray(input_array)
    else:
        ds_data = gdal.Open(output_str, gdal.GA_Update)
        ds_data.WriteArray(input_array, xoff=0, yoff=block_row)

    ds_data = None
    del ds_data