# NETWORK MAIN CLASSES AND METHODS DEFINITION RELATED TO IT.

class Networking:

    def __init__(self):

        pass

    def lookup_mapping(self, sinr_dB):

        pass

    def pat_loss(self, pBS, pMS, hroof, phi, wr, fMHz, wb, city='medium'):

        # Compute the path loss.
        # INPUTS:
        # - pBs     -->     position of the base station (x,y,z);
        # - pMS     -->     position of the mobile station (x,y,z);
        # - hroof   -->     buildings height;
        # - phi     -->     orientation angle between the direction of propagation and street axix [degrees];
        # - wr      -->     width of the streets [m];
        # - fMhz    -->     carrier frequency [MHz].
        # OUTPUT:
        # - path_loss (Nlink,Nf). 

        # cost231 uses the 'Walfish-Ikegami model' to compute the PathLoss:
        path_loss = cost231(pBS, pMS, hroof, phi, wr, fMHz, wb, city)

        return path_loss

    def from_pwr_to_dB(self, pwr, dimension_factor):
        # Convert power into dB;
        # 'dimension_factor' is the factor to convert a multiple of [W] in [W].  

        pwr_dB = 10*log10(pwr)/dimension_factor

        return pwr_dB

    def from_dB_to_pwr(self, pwr_db, dimension_factor):
        # Convert dB into power [W];
        # 'dimension_factor' is the factor to convert a multiple of [W] in [W].

        pwr = pow(10, pwr_db/10)/dimension_factor

        return pwr

    def receiving_power(self, Ptx_dBW, gain_dB, path_loss_dB):
        # Takes in input the transmissing power [dBW], the gain [dB] and the path loss [dB];
        # it returns the receiving power [dBW]. 

        Prx_dBW = Ptx_dBm + gain_dB - path_loss_dB
        
        return Prx_dBW

    def noise_power_in_bandwith(self, figure_noise_dB, bw_Hz):
        # Takes in input the figure_noise [dB] of the bandwith [Hz];
        # it returns the noise power [dBm] contained in bandwith.

        N_dB = figure_noise_dB - 174 + 10*log10(bw_Hz)

        # 1000 [W] = 60 [dBm], then '+60' is added to get [dBm]/[Hz]: 
        N_dBm = N_db + 60

    def signal_to_noise_ratio(self, Prx_W, Pnoise_W):
        # Returns SNR (signal-to-noise Ratio).

        SNR = Prx/Pnoise

        return SNR


