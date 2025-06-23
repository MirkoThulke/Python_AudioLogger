import unittest
import os
import multiprocessing
import sys
import numpy as np
from scipy.signal import correlate
from scipy.signal import cheby2, freqz, sosfilt, sosfreqz
import matplotlib.pyplot as plt


# Add the parent directory (i.e., one level up from `scripts/`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Programming\eclipse_workspace\Python_AudioLogger")))

import Python_AudioLogger
from Python_AudioLogger import create_shared_resource_manager
from Python_AudioLogger import create_process_local_common_datadictionary_definition
from Python_AudioLogger import create_shared_memory_resources
from Python_AudioLogger import apply_low_pass
from Python_AudioLogger import nyquist, normal_cutoff, STOPBAND_ATTEN, CUTOFF, RATE, ORDER


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


def power_spectral_sensity_psd(spectrum, len, signal_mask, freqs):
    
    spectrum = spectrum.astype(np.float64)
    psd = (1/len) * (np.abs(spectrum)**2)
    return psd

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
        result_H_cutoff     = False
        result_H_dc         = False
        
        
        # Compute the frequency response

        sos = cheby2(N=ORDER, rs=STOPBAND_ATTEN, Wn=CUTOFF, btype='low', fs=RATE, output='sos')
        
        # Compute frequency response
        frequencies, h = sosfreqz(sos, worN=1024, fs=RATE)  # w: frequency axis in Hz, h: complex gain
        
        
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
        if H_cutoff < -STOPBAND_ATTEN+1 :
            result_H_cutoff = True
            
        # check if attentuation is ZERO at 0 Hz
        if -0.5 < H_dc < 0.5 :
            result_H_dc = True
        
        
        result = result_H_cutoff and result_H_dc
        
        self.assertTrue(result, "Expected result to be True")
        
        

    def test_Unit_02_Check_LowPass_Check_withSinus(self):
        result_gain_100hz   = False
        result_gain_1000hz  = False
        

        self.sinus_100hz_s16 = generate_sine_wave_bytes(frequency=100, duration=3.0, amplitude=1.0)
        self.sinus_1000hz_s16 = generate_sine_wave_bytes(frequency=1000, duration=3.0, amplitude=1.0)
        
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)
        
        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        lowpass_array_1000hz   = apply_low_pass(self.data_dictionary)
        

        unfiltered_energy_100hz     = signal_energy(self.sinus_100hz_s16)
        lowpass_energy_100hz        = signal_energy(lowpass_array_100hz )
        unfiltered_energy_1000hz    = signal_energy(self.sinus_1000hz_s16)
        lowpass_energy_1000hz       = signal_energy(lowpass_array_1000hz )
        
        print(f"Average power: unfiltered_energy_100hz : {unfiltered_energy_100hz}")
        print(f"Average power: lowpass_energy_100hz : {lowpass_energy_100hz}")
        print(f"Average power: unfiltered_energy_1000hz : {unfiltered_energy_1000hz}")
        print(f"Average power: lowpass_energy_1000hz : {lowpass_energy_1000hz}")
        
        
        lowpass_gain_100hz =   10 * np.log10(lowpass_energy_100hz / unfiltered_energy_100hz)
        lowpass_gain_1000hz =  10 * np.log10(lowpass_energy_1000hz / unfiltered_energy_1000hz)
        
        print(f"lowpass_gain_100hz: {lowpass_gain_100hz}")
        print(f"lowpass_gain_1000hz: {lowpass_gain_1000hz}")
        
        # expected gain to be close to zero+ tolerance
        if -1.5 <= lowpass_gain_100hz <= 1.5 :
            result_gain_100hz = True
            
        # expected gain to be arround expected attentuation + tolerance
        if lowpass_gain_1000hz <= -(STOPBAND_ATTEN -30) :
            result_gain_1000hz = True
        
        result = result_gain_100hz and result_gain_1000hz
        
        self.assertTrue(result, "Expected result to be True")

    
    
    def test_Unit_03_Check_LowPass_Check_OutPut_AudioIntegrity(self):
        result = False
        FREQUENCY = 100
        
        self.sinus_100hz_s16 = generate_sine_wave_bytes(frequency=FREQUENCY, duration=2, amplitude=1.0)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)
        
        energy           =  signal_energy(self.sinus_100hz_s16)
        energy_filtered  =  signal_energy(lowpass_array_100hz)
        
        
        # Perform FFT on the audio signal
        fft_signal              = np.fft.fft(self.sinus_100hz_s16)
        fft_signal_filtered     = np.fft.fft(lowpass_array_100hz)
        
        # Compute the corresponding frequencies
        frequencies = np.fft.fftfreq(len(fft_signal), d=1/RATE)
        frequencies_pos = frequencies > 0
        


        # Get the magnitude of the FFT and normalise to max value 
        fft_magnitude                   = np.abs(fft_signal)
        fft_psd                         = power_spectral_sensity_psd(fft_magnitude, len(self.sinus_100hz_s16), frequencies_pos, frequencies )

        fft_magnitude_filtered          = np.abs(fft_signal_filtered)
        fft_psd_filtered                = power_spectral_sensity_psd(fft_magnitude_filtered, len(self.sinus_100hz_s16), frequencies_pos, frequencies )


        
        plt.figure()
        plt.semilogx(frequencies[frequencies_pos], fft_psd_filtered[frequencies_pos])
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('power_spectral_sensity')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig("power_spectral_sensity.png", dpi=300)
        
        

        denominator = energy_filtered - energy
        
        if denominator <= 0 :
            print("⚠️ Invalid SNA calculation: denominator <= 0")
            SNA_dB = -np.inf  # or set to None or NaN
        else :
            SNA_dB = 10 * np.log10(energy / (energy_filtered - energy))
            
        print(f"energy: {energy}")
        print(f"energy_filtered: {energy_filtered}")
        print(f"SNA_dB: {SNA_dB}")
        
        if SNA_dB > 30 :
            result = True
        
        
        self.assertTrue(result, "Expected result to be True")
    


# to call from comand line :
# alternative : use > python -m unittest test_unit_01.py
# ✅ RUN TESTS
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional, but good for Windows
    unittest.main()