import unittest
import os
import multiprocessing
import sys
import numpy as np
from numpy import pi
from scipy.signal import correlate
from scipy.signal import freqz
import matplotlib.pyplot as plt
import pandas as pd
import wave


# Add the parent directory as relative path to import Python_AudioLogger and endolith_weighting_filters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import Python_AudioLogger
from Python_AudioLogger import create_shared_resource_manager
from Python_AudioLogger import create_process_local_common_datadictionary_definition
from Python_AudioLogger import create_shared_memory_resources
from Python_AudioLogger import RATE, SAMPLE_SIZE, CHANNELS, CHUNK, STOPBAND_ATTEN, CUTOFF, LOWPASS_INIT_STATE
from Python_AudioLogger import apply_low_pass, apply_a_weighting
from endolith_weighting_filters import A_weight, A_weighting


# A-Weighting Filter ATTENTUATION
AWEIGHT_DB_6_3	    =	-19.0
AWEIGHT_DB_8	    =	-77.6
AWEIGHT_DB_10	    =	-70.4
AWEIGHT_DB_12_5	    =	-63.6
AWEIGHT_DB_16	    =	-56.4
AWEIGHT_DB_20	    =	-50.4
AWEIGHT_DB_25	    =	-44.8
AWEIGHT_DB_31_5	    =	-39.5
AWEIGHT_DB_40	    =	-34.5
AWEIGHT_DB_50	    =	-30.3
AWEIGHT_DB_63	    =	-26.2
AWEIGHT_DB_80	    =	-22.4
AWEIGHT_DB_100	    =	-19.1
AWEIGHT_DB_125	    =	-16.2
AWEIGHT_DB_160	    =	-13.2
AWEIGHT_DB_200	    =	-10.8
AWEIGHT_DB_250	    =	-8.7
AWEIGHT_DB_315	    =	-6.6
AWEIGHT_DB_400	    =	-4.8
AWEIGHT_DB_500	    =	-3.2
AWEIGHT_DB_630	    =	-1.9
AWEIGHT_DB_800	    =	-0.8
AWEIGHT_DB_1000	    =	0
AWEIGHT_DB_1250	    =	0.6
AWEIGHT_DB_1600	    =	1
AWEIGHT_DB_2000	    =	1.2
AWEIGHT_DB_2500	    =	1.3
AWEIGHT_DB_3150	    =	1.2
AWEIGHT_DB_4000	    =	1
AWEIGHT_DB_5000	    =	0.6
AWEIGHT_DB_6300	    =	-0.1
AWEIGHT_DB_8000	    =	-1.1
AWEIGHT_DB_10000    =	-2.5
AWEIGHT_DB_12500	=	-4.3
AWEIGHT_DB_16000	=	-6.7
AWEIGHT_DB_20000	=	-9.3

# Pytest section ########
PASSED      = True
FAILED      = False

result  = FAILED 
#########################

#########################
# Pytest calls all functions starting with "test_" automatically
# hence, functions to be called by pytest MUST start with "test_"

# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared


# generate a sinus signal at 1000Hz in 48000 hz pcm format in  int16 :
def generate_sine_wave_bytes(frequency=100, sample_rate=48000, duration=1.0, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Scale to int16
    samples_int16 = np.int16(samples * 32767)

    # Return as raw bytes
    return samples_int16, t


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


def calculate_snr(signal, noisy_signal):
    signal          = signal.astype(np.float32)
    noisy_signal    = noisy_signal.astype(np.float32)
    
    noise = noisy_signal - signal

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    epsilon = 1e-10  # Small constant to avoid divide-by-zero or log(0)

    snr = signal_power / (noise_power + epsilon)
    snr_db = 10 * np.log10(snr + epsilon)  # Ensure you also avoid log(0) here
    
    return snr_db


# ✅ TEST CLASS: always define outside `if __name__ == "__main__"`
class UnitTest_AWeighting(unittest.TestCase):
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

        
    print("\n")
    print("\n")
    
    # a) Plot the frequency response for visual checks
    # b) Check if the attenuation within the passband is within acceptable limits.
    # c) Print filter gain ata specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    def test_Unit_01_Check_AWeighted_Check_TransferFunction(self):
        result_H_100Hz          = False
        result_H_1000Hz         = False
        result_H_1600Hz         = False
        
        # Compute frequency response
        b, a = A_weighting(RATE)
        frequencies = np.geomspace(10, RATE/4, 1000)
        w = 2*pi * frequencies / RATE
        w, h = freqz(b, a, w)
           
        
        # Plot on log x-axis
        plt.figure()
        plt.figure(figsize=(10, 6))
        plt.semilogx(frequencies, 20 * np.log10(np.abs(h)), 'b')
        plt.title("A-Weighting Filter Frequency Response")
        plt.xlabel("Frequency [Hz] (log scale)")
        plt.ylabel("Amplitude [dB]")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axvline(100, color='blue', linestyle='--', label='100 Hz Frequency')
        plt.axvline(1000, color='green', linestyle='--', label='1000 Hz Frequency')
        plt.axvline(1600, color='red', linestyle='--', label='1600 Hz Frequency')
        plt.axhline(AWEIGHT_DB_100, color='black', linestyle='--', label=f"{AWEIGHT_DB_100:.2f} dB")
        plt.axhline(AWEIGHT_DB_1000, color='black', linestyle='--', label=f"{AWEIGHT_DB_1000:.2f} dB")
        plt.axhline(AWEIGHT_DB_1600, color='black', linestyle='--', label=f"{AWEIGHT_DB_1600:.2f} dB")
        plt.legend()
        plt.savefig("filter_response_aweight.png", dpi=300)
        plt.close()
        
        
        # Find the index closest to 100Hz  
        idx_100Hz = np.argmin(np.abs(frequencies - 100))

        # Find the index closest to 1000Hz  
        idx_1000Hz = np.argmin(np.abs(frequencies - 1000))
    
        # Find the index closest to 16000Hz  
        idx_1600Hz = np.argmin(np.abs(frequencies - 1600))
        
        H_100Hz_dB          = 20 * np.log10(np.abs(h[idx_100Hz]))
        H_1000Hz_dB         = 20 * np.log10(np.abs(h[idx_1000Hz]))
        H_1600Hz_dB         = 20 * np.log10(np.abs(h[idx_1600Hz]))

        # check if attentuation is as expected at the specified cutt off frenquency
        if AWEIGHT_DB_100-0.1 < H_100Hz_dB < AWEIGHT_DB_100+0.1 :
            result_H_100Hz = True
        
        # check if attentuation is as expected at the specified cutt off frenquency
        if AWEIGHT_DB_1000-0.1 < H_1000Hz_dB < AWEIGHT_DB_1000+0.1 :
            result_H_1000Hz = True
            
        # check if attentuation is as expected at the specified cutt off frenquency
        if AWEIGHT_DB_1600-0.1 < H_1600Hz_dB < AWEIGHT_DB_1600+0.1 :
            result_H_1600Hz = True
        
                # Put into a DataFrame
        df = pd.DataFrame({
            'frequencies': frequencies,
            'h': h,
            'frequencies[idx_100Hz]': frequencies[idx_100Hz],
            'frequencies[idx_1000Hz]': frequencies[idx_1000Hz],
            'frequencies[idx_16000Hz]': frequencies[idx_1600Hz],
            'H_1000Hz_dB': H_100Hz_dB,
            'H_1000Hz_dB': H_1000Hz_dB,
            'H_1000Hz_dB': H_1600Hz_dB
        })

        # Write to Excel
        df.to_excel('test_Unit_aweighted_01.xlsx', index=False)
        
        
        print(f"test_Unit_aweighted_01:")
        print(f"frequencies[idx_100Hz]: {frequencies[idx_100Hz]}")
        print(f"frequencies[idx_1000Hz]: {frequencies[idx_1000Hz]}")
        print(f"frequencies[idx_1600Hz]: {frequencies[idx_1600Hz]}")
        print(f"H_100Hz_dB: {H_100Hz_dB:.2f}")
        print(f"H_1000Hz_dB: {H_1000Hz_dB:.2f}")
        print(f"H_16000Hz_dB: {H_1600Hz_dB:.2f}")
        print(f"result_H_100Hz: {result_H_100Hz}")
        print(f"result_H_1000Hz: {result_H_1000Hz}")
        print(f"result_H_16000Hz: {result_H_1600Hz}")
        print("-----------------\n")
        print("\n")
        
        
        result = result_H_100Hz and result_H_1000Hz and result_H_1600Hz
        
        self.assertTrue(result, "test_Unit_aweighted_01 : Expected result to be True")
        

        

    # a) Check if the minimum attention within the stop band matchs the filter parameter specification.
    # b) Check if the attenuation within the passband is within acceptable limits.
    # c) Print filter gain ata specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    def test_Unit_02_Check_AWeighted_Check_withSinus(self):

        result_gain_100hz   = False
        result_gain_1000hz  = False
        result_filter_init  = False
        
            
            
        test_chunk_duration = CHUNK / RATE


        self.sinus_100hz_s16,time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_1000hz_s16,time  = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        
        aweighted_array_100hz   = apply_a_weighting(self.data_dictionary)
        
        
        
        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        
        aweighted_array_1000hz   = apply_a_weighting(self.data_dictionary)
        
        
        # Calculate energy
        unfiltered_energy_100hz     = signal_energy(self.sinus_100hz_s16)
        aweighted_energy_100hz        = signal_energy(aweighted_array_100hz )
        unfiltered_energy_1000hz    = signal_energy(self.sinus_1000hz_s16)
        aweighted_energy_1000hz       = signal_energy(aweighted_array_1000hz )
        
        # Calculate gain
        aweighted_gain_100hz =   10 * np.log10(aweighted_energy_100hz / unfiltered_energy_100hz)
        aweighted_gain_1000hz =  10 * np.log10(aweighted_energy_1000hz / unfiltered_energy_1000hz)
        
        
        # expected gain to be inside tolerance
        if -20 <= aweighted_gain_100hz <= -18 :
            result_gain_100hz = True
        
                # expected gain to be inside tolerance
        if -1 <= aweighted_gain_1000hz <= 1:
            result_gain_1000hz = True
             
        
        
        
        # Put into a DataFrame
        df = pd.DataFrame({
            'self.sinus_100hz_s16': self.sinus_100hz_s16,
            'self.sinus_1000hz_s16': self.sinus_1000hz_s16,
            'lowpass_array_1000hz': aweighted_array_1000hz,
            'time': time
        })

        # Write to Excel
        df.to_excel('test_Unit_02.xlsx', index=False)
    
    
        print(f"test_Unit_02:\n")
        print(f"aweighted_gain_100hz: {aweighted_gain_100hz:.2f}")
        print(f"aweighted_gain_1000hz: {aweighted_gain_1000hz:.2f}") 
        print(f"result_gain_100hz: {result_gain_100hz}")
        print(f"result_gain_1000hz: {result_gain_1000hz}")
        print("-----------------\n")
        print("\n")
        
        result =  result_gain_100hz and result_gain_1000hz
        
        self.assertTrue(result, "test_Unit_aweighted_02 : Expected result to be True")
    
    
    # a) Calculate SNR for a specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    # b) Check if the specified filter attentuation is met
    def test_Unit_03_Check_AWeighted_CheckAudioOutput_OneChunk(self):
        result_SNA_oneChunk_100hz_dB = False
        result_SNA_oneChunk_1000hz_dB = False
        
 
        test_chunk_duration = CHUNK / RATE
        
        
        self.sinus_100hz_s16, time  = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_1000hz_s16, time = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_zero_s16, time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1e-10)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16

        lowpass_array_100hz   = apply_a_weighting(self.data_dictionary)


        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16

        lowpass_array_1000hz   = apply_a_weighting(self.data_dictionary)
        

        SNA_oneChunk_100hz_dB    = calculate_snr(self.sinus_100hz_s16, lowpass_array_100hz)
        SNA_oneChunk_1000hz_dB   = calculate_snr(self.sinus_zero_s16, lowpass_array_1000hz)
        
        if 1.0 >=SNA_oneChunk_100hz_dB >= -1.0 :
            result_SNA_oneChunk_100hz_dB = True
        if SNA_oneChunk_1000hz_dB <= -80.0 :
            result_SNA_oneChunk_1000hz_dB = True
        

          
        # Put into a DataFrame
        df = pd.DataFrame([{
            'SNA_oneChunk_100hz_dB': SNA_oneChunk_100hz_dB,
            'SNA_oneChunk_1000hz_dB': SNA_oneChunk_1000hz_dB
        }])

        # Write to Excel
        df.to_excel('test_Unit_03.xlsx', index=False)


        print(f"test_Unit_03:\n")       
        print(f"SNA_oneChunk_100hz_dB: {SNA_oneChunk_100hz_dB:.2f}")
        print(f"SNA_oneChunk_1000hz_dB: {SNA_oneChunk_1000hz_dB:.2f}")       
        print(f"result_SNA_oneChunk_100hz_dB: {result_SNA_oneChunk_100hz_dB}")
        print(f"result_SNA_oneChunk_1000hz_dB: {result_SNA_oneChunk_1000hz_dB}")  
        print("-----------------\n")
        print("\n")

        result = result_SNA_oneChunk_100hz_dB and result_SNA_oneChunk_1000hz_dB
         
        self.assertTrue(result, "test_Unit_03 : Expected result to be True")
    

    # a) Treat signal chunkwise, to test for artefacts due to errors in filter state losses etc.
    # b) Calculate SNR again over complete joint signal
    # c) Plot unfiltered and filtered sinus and overlay signals
    # d) Save wave files to check by licening if noise is present in the filtered signals
    def test_Unit_04_Check_AWeighted_CheckAudioOutput_appendChunks(self):
        result_SNA_100hz_dB     = False
        result_SNA_1000hz_dB    = False
        
        chunk_duration = CHUNK / RATE
        
        
        test_duration = 4.0 # seconds
        test_number_chunks = int(test_duration / chunk_duration)
        
        
        self.sinus_100hz_s16, time  = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_duration, amplitude=1.0)
        self.sinus_1000hz_s16, time = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_duration, amplitude=1.0)
        self.sinus_zero_s16, time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_duration, amplitude=1e-10)
        
        # Simulate the chunk wise low pass filtering in order to detect noise due to transients etc.


        lowpass_chunks      = []
        filtered_chunk      = []
        i = 0
        # initialse lowpass filter state
        self.data_dictionary['lowpass_filter_state']    = LOWPASS_INIT_STATE
        
        for i in range(0, len(self.sinus_100hz_s16), CHUNK):        
            # Process the chunk
            self.data_dictionary['audio_data']  = self.sinus_100hz_s16[i:i+CHUNK]
            filtered_chunk                      = apply_low_pass(self.data_dictionary)
            lowpass_chunks.append(filtered_chunk)
        
        # Join all filtered chunks into one array
        lowpass_array_100hz = np.concatenate(lowpass_chunks)



        lowpass_chunks          = []
        filtered_chunk          = []
        i = 0
        for i in range(0, len(self.sinus_1000hz_s16), CHUNK):        
            # Process the chunk
            self.data_dictionary['audio_data']  = self.sinus_1000hz_s16[i:i+CHUNK] 
            filtered_chunk                      = apply_low_pass(self.data_dictionary)
            lowpass_chunks.append(filtered_chunk)
        
        # Join all filtered chunks into one array
        lowpass_array_1000hz = np.concatenate(lowpass_chunks)
        
        
        SNA_100hz_dB    = calculate_snr(self.sinus_100hz_s16, lowpass_array_100hz[:len(self.sinus_100hz_s16)])
        SNA_1000hz_dB   = calculate_snr(self.sinus_zero_s16, lowpass_array_1000hz[:len(self.sinus_1000hz_s16)])
        
        if SNA_100hz_dB >= -5 :
            result_SNA_100hz_dB = True
        if SNA_1000hz_dB <= -80.0 :
            result_SNA_1000hz_dB = True
        

        
        
        # Put into a DataFrame
        df = pd.DataFrame([{
            'self.sinus_100hz_s16': self.sinus_100hz_s16,
            'self.sinus_1000hz_s16': self.sinus_1000hz_s16,
            'self.sinus_zero_s16': self.sinus_zero_s16,
            'lowpass_array_100hz': lowpass_array_100hz,
            'lowpass_array_1000hz': lowpass_array_1000hz,                        
            'SNA_100hz_dB': SNA_100hz_dB,
            'SNA_1000hz_dB': SNA_1000hz_dB
        }])

        # Write to Excel
        df.to_excel('test_Unit_04.xlsx', index=False)


        with wave.open('sinus_100hz_s16_unfiltered.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(self.sinus_100hz_s16.tobytes())


        with wave.open('lowpass_array_100hz.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(lowpass_array_100hz.tobytes())


        with wave.open('lowpass_array_1000hz.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(lowpass_array_1000hz.tobytes())



        plt.figure()
        plt.plot(time[:8000], self.sinus_100hz_s16[:8000], color='blue', label='sinus_100hz_s16')
        plt.plot(time[:8000], lowpass_array_100hz[:8000], color='green', label='lowpass_array_100hz')
        plt.plot(time[:8000], lowpass_array_1000hz[:8000], color='orange', label='lowpass_array_1000hz')
        plt.xlabel('time')
        plt.legend()
        plt.ylabel(f"pcm. LowPass Filtered with CutOff freq.: {CUTOFF} ")
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.savefig("self.sinus100hz_LowPass100hz_LoPass1000hz_s16.png", dpi=600)
        plt.close()


        print(f"test_Unit_04:\n")
        print(f"Audio saved as {'sinus_100hz_s16_unfiltered.wav'}\n")
        print(f"Audio saved as {'lowpass_filtered_100hz.wav'}\n")
        print(f"Audio saved as {'lowpass_filtered_1000hz.wav'}\n")
        print(f"SNA_100hz_dB: {SNA_100hz_dB:.2f}")
        print(f"SNA_1000hz_dB: {SNA_1000hz_dB:.2f}")
        print(f"result_SNA_100hz_dB: {result_SNA_100hz_dB}")
        print(f"result_SNA_1000hz_dB: {result_SNA_1000hz_dB}")
        print("-----------------\n")
        print("\n")
        
        result = result_SNA_100hz_dB and result_SNA_1000hz_dB
        
        self.assertTrue(result, "test_Unit_04 : Expected result to be True")



# to call from comand line :
# alternative : use > python -m unittest test_unit_01.py
# ✅ RUN TESTS
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional, but good for Windows
    unittest.main()
    
    result = PASSED # pytest result
    
    sys.exit(result)  # this return code goes back to Jenkins
