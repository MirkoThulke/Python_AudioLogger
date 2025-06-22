import unittest
import os
import multiprocessing
import sys
import numpy as np
from scipy.signal import correlate
from scipy.signal import cheby2, freqz
import matplotlib.pyplot as plt


# Add the parent directory (i.e., one level up from `scripts/`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Programming\eclipse_workspace\Python_AudioLogger")))

import Python_AudioLogger
from Python_AudioLogger import create_shared_resource_manager
from Python_AudioLogger import create_process_local_common_datadictionary_definition
from Python_AudioLogger import create_shared_memory_resources
from Python_AudioLogger import apply_low_pass
from Python_AudioLogger import cheby2_b, cheby2_a, nyquist, normal_cutoff, STOPBAND_ATTEN, CUTOFF, RATE


# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared



# generate a sinus signal at 1000Hz in 48000 hz raw byte pcm coded format int16 :
def generate_sine_wave_bytes(frequency=1000, sample_rate=48000, duration=1.0, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Scale to int16
    samples_int16 = np.int16(samples * 32767)

    # Return as raw bytes
    return samples_int16


def signal_energy(signal):
    
    signal = signal.astype(np.float64)
    energy = np.sum(np.abs(signal)**2)
    return energy


def signal_rms(signal):
    
    signal = signal.astype(np.float64)
    rms = np.sqrt(np.mean(signal**2))
    return rms


def spectrum_energy(spectrum):
    
    spectrum = spectrum.astype(np.float64)
    energy = (np.abs(spectrum)**2)/len(spectrum)
    return energy

# ✅ TEST CLASS: always define outside `if __name__ == "__main__"`
class UnitTest_LowPass(unittest.TestCase):
    def setUp(self):
        self.manager = create_shared_resource_manager()
        self.data_dictionary = create_process_local_common_datadictionary_definition(self.manager)

        (
            self._device_index,
            self.chunk_index_i,
            self.is_recording,
            self.is_logging,
            self.is_lowpass,
            self.system_calibration_factor_94db,
            self.frames,
            self.frames_filtered,
            self.chunk_noise_list_index,
            self.chunk_noise_list_spl
        ) = create_shared_memory_resources(self.manager)


    def tearDown(self):
        self.manager.shutdown()


    def test_Unit_01_Check_LowPass_Check_TransferFunction(self):
        
        # Compute the frequency response
        w, h = freqz(cheby2_b, cheby2_a, worN=8000)
        
        frequencies = w * nyquist / np.pi  # Convert from rad/sample to Hz

        # Plot on log x-axis
        plt.figure()
        plt.figure(figsize=(10, 6))
        plt.semilogx(frequencies, 20 * np.log10(np.abs(h)), 'b')
        plt.title("Chebyshev Type II Lowpass Filter Frequency Response (Log X-Axis)")
        plt.xlabel("Frequency [Hz] (log scale)")
        plt.ylabel("Amplitude [dB]")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axvline(CUTOFF, color='red', linestyle='--', label='Cutoff Frequency')
        plt.axhline(-STOPBAND_ATTEN, color='green', linestyle='--', label='Stopband Attenuation')
        plt.legend()
        plt.savefig("filter_response_lowpass.png", dpi=300)
        
        # Find the index closest to normal_cutoff
        idx_cutoff = np.argmin(np.abs(frequencies - CUTOFF))
        
        H_cutoff        = 20 * np.log10(np.abs(h[idx_cutoff]))
        H_dc            = 20 * np.log10(np.abs(h[0]))
        print(f"H_cutoff: {H_cutoff}")
        print(f"H_dc: {H_dc}")
        
        # check if attentuation is as expected at the specified cutt off frenquency
        if -STOPBAND_ATTEN-1 < H_cutoff < -STOPBAND_ATTEN+1 :
            result_H_cutoff = True
        # check if attentuation is ZERO at 0 Hz
        if -0.5 < H_dc < 0.5 :
            result_H_dc = True
        
        
        result = result_H_cutoff and result_H_dc
        
        self.assertTrue(result, "Expected result to be True")
        
        

    def test_Unit_02_Check_LowPass_Check_withSinus(self):
        result_gain_100hz   = False
        result_gain_1000hz  = False
        
        self.sinus_1000hz_s16 = generate_sine_wave_bytes(frequency=1000, duration=3.0, amplitude=1.0)
        self.sinus_100hz_s16 = generate_sine_wave_bytes(frequency=100, duration=3.0, amplitude=1.0)
        
        
        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        lowpass_array_1000hz   = apply_low_pass(self.data_dictionary)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)

        unfiltered_energy_100hz     = signal_energy(self.sinus_100hz_s16)
        lowpass_energy_100hz        = signal_energy(lowpass_array_100hz )
        unfiltered_energy_1000hz    = signal_energy(self.sinus_1000hz_s16)
        lowpass_energy_1000hz       = signal_energy(lowpass_array_1000hz )
        
        #print(f"Average power: unfiltered_energy_200hz : {unfiltered_energy_100hz}")
        #print(f"Average power: lowpass_energy_200hz : {lowpass_energy_100hz}")
        #print(f"Average power: unfiltered_energy_1000hz : {unfiltered_energy_1000hz}")
        #print(f"Average power: lowpass_energy_1000hz : {lowpass_energy_1000hz}")
        
        
        lowpass_gain_100hz =   10 * np.log10(lowpass_energy_100hz / unfiltered_energy_100hz)
        lowpass_gain_1000hz =  10 * np.log10(lowpass_energy_1000hz / unfiltered_energy_1000hz)
        
        print(f"lowpass_gain_100hz: {lowpass_gain_100hz}")
        print(f"lowpass_gain_1000hz: {lowpass_gain_1000hz}")
        
        if -1.0 <= lowpass_gain_100hz <= 1.0 :
            result_gain_100hz = True  # Replace with real test logic
            
        if lowpass_gain_1000hz <= -40.0 :
            result_gain_1000hz = True  # Replace with real test logic
        
        result = result_gain_100hz and result_gain_1000hz
        
        self.assertTrue(result, "Expected result to be True")

    
    
    def test_Unit_03_Check_LowPass_Check_OutPut_AudioIntegrity(self):
        result = False
        FREQUENCY = 100
        self.sinus_100hz_s16 = generate_sine_wave_bytes(frequency=FREQUENCY, duration=0.5, amplitude=1.0)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)
        
        
        # Perform FFT on the audio signal
        fft_signal              = np.fft.fft(self.sinus_100hz_s16)
        fft_signal_filtered     = np.fft.fft(lowpass_array_100hz)

    
        # Get the magnitude of the FFT and normalise to max value 
        fft_magnitude                   = np.abs(fft_signal)
        fft_energy                      = spectrum_energy(fft_magnitude)

        fft_magnitude_filtered          = np.abs(fft_signal_filtered)
        fft_energy_filtered             = spectrum_energy(fft_magnitude_filtered )


        fft_energy_error                = fft_energy - fft_energy_filtered
        fft_energy_error_norm           = fft_energy_error / max(fft_energy)
        
        # Compute the corresponding frequencies
        frequencies = np.fft.fftfreq(len(fft_signal), d=1/RATE)
        
        pos = frequencies > 0
        
        plt.figure()
        plt.semilogx(frequencies[pos], fft_energy_error_norm[pos], label='Difference')
        
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalised Energy Spectrum Error')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig("Normalized_Energy_Spectrum_Error.png", dpi=300)
        
        # Find the index closest to normal_cutoff
        idx_frequency = np.argmin(np.abs(FREQUENCY - CUTOFF))
        
        # Check if the error at the simulated frequency is the maximum error.
        # If not, then this can only be noise 
        print(f"fft_energy_error_norm[idx_frequency]: {fft_energy_error_norm[idx_frequency]}")
        print(f"max(fft_energy_error_norm)]: {max(fft_energy_error_norm)}")       
        if fft_energy_error_norm[idx_frequency] >= max(fft_energy_error_norm):
            result = True
        
        
        self.assertTrue(result, "Expected result to be True")
    


# to call from comand line :
# alternative : use > python -m unittest test_unit_01.py
# ✅ RUN TESTS
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional, but good for Windows
    unittest.main()