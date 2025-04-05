'''
# -----------------------------------------------------------------------------
# Author: MIRKO THULKE 
# Copyright (c) 2025, MIRKO THULKE
# All rights reserved.
#
# Date: 2025, VERSAILLES, FRANCE
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING
# FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# -----------------------------------------------------------------------------
'''


'''
important commands : 
cmd> python -m pip install --upgrade pip
cmd> pip install pyaudio
cmd> pip install numpy
cmd> where python
cmd> python --version
cmd> python3 --version
cmd> pip install --upgrade -r requirements.txt
cmd> pip freeze > requirements.txt
cmd> pip list --outdated
'''

import pyaudio
import numpy as np
import wave
import wx # click button GUI
import matplotlib.pyplot as plt
import configparser
import threading
import time
import subprocess
import datetime
import os
import scipy.signal as signal
import sympy
import librosa
from endolith_weighting_filters import A_weight


# Behringer UMC control panel settings :
# ASIO buffer size 512   
# Sampling rate 48kHz
# Format : 4 Channel 16 bits  


# Set parameters for audio input
FORMAT = pyaudio.paInt16  # Format for the audio
CHANNELS = 1  # Mono audio (1 channel)
RATE = 48000  # Sampling rate (samples per second)
CHUNK = 1024  # Number of frames per buffer (size of each audio chunk)
DEVICE_INDEX = None  # Set to the correct device index if you have multiple devices
CHUNK_SEC = CHUNK/RATE # Chunk duration in seconds
WAVE_DT_SEC = 1 # Delta time duration before and after noise event, that will be added to wave output
CHUNK_DNUM = int(WAVE_DT_SEC/CHUNK_SEC) # number of chunks to be added before and after noise event, that will be added to wave output


REFERENCE_PRESSURE = 20e-6  # in Pa. Reference pressure in Pa (20 µPa)
MIC_SENSITIVITY = 15 # 15mV Sensitivity (mV / Pa) of the Behringer ECM8000 (for better understanding only)
MIC_SPL_MAX_DB = 120   # in dB. virtual SPL max. value that the microphone can measure
MIC_PA_MAX = 20     # in Pa. Maximum Number of Pa that the mcicrophone can measure (120dB converted to Pascal)
MIC_MAX_MVOLT = MIC_PA_MAX * MIC_SENSITIVITY # maximum mV that the microphone can measure (corresponds to 120dB, 20Pa)

PRE_AMP_GAIN_DB = 45  # in dB. Assupmption that the Gain poti on the UMC is on 3h00. Virtuel Pre-Amp Gain Factor for better understanding only
PRE_AMP_GAIN_LIN = np.power(10, (PRE_AMP_GAIN_DB / 20) ) # Converted to linear scale.

system_calibration_factor_94db = 1 # Microphone calibration factor to obtain 94dB at 1kHz. Default value 1
CALIB_ITERATION_LENGTH = 50 # 50 chunks are checked during calibration
MAX_INT16 = np.iinfo(np.int16).max

# dB(A) Threshold for indoor noise
# NIGHT : 22h00 - 06h00
SPL_MAX_DAY_DBA     = 50
SPL_MAX_NIGH_DBA    = 35



# Set parameters for audio outut
RECORD_SECONDS = 4        # Duration of the recording in seconds
OUTPUT_FILENAME = "output.wav"  # Output WAV file
OUTPUT_NOISE_FILENAME = "Bruit" #Output filename prefix for logged noise events
OUTPUT_FILE_DIRECTORY = "audio_logfiles"

#Global Variables #######################################

# Initialize PyAudio
p                                   = pyaudio.PyAudio()

# Device list user input 
_device_index                       = 0

# Persist Settings. Create config object.
config                              = configparser.ConfigParser()

# flag to track recoding state
is_recording                        = False
# flag to track logging state
is_logging                          = False

# Open stream to read audio from the microphone

stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index = _device_index ,
    frames_per_buffer=CHUNK)

print("Audio stream opened")


# audio data extraceted from chunk in np format.
audio_data                          = np.array([])
a_weighted_signal                   = np.array([])
audio_data_pcm_abs                  = np.array([])
audio_data_mV                       = np.array([])
audio_data_mV_calib                 = np.array([])
audio_data_pressurePa               = np.array([])
audio_data_pressurePa_square        = np.array([])
audio_data_pressurePa_squareMean    = np.array([])
audio_data_pressurePa_rms           = np.array([])
audio_data_pressurePa_rms_calib     = np.array([])
audio_data_pressurePa_spl           = np.array([])
audio_data_max_pcm_value            = 0

#for output wave file creation / all chunks, complete measurement
frames                              = []

chunk_index_i                       = 0
chunk_noise_list_index              = []
chunk_noise_list_spl                = []


# Global Funtion definitions ##########################################
def get_commit_version():
    try:
        # Run 'git rev-parse HEAD' to get the commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return "Not a git repository or error retrieving commit."
 
def func_check_devices():
    global p
    
    i=0
    
    # List available devices
    print("Available devices:")
    for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")



def func_on_button_setDevices_click(frame):
    global _device_index 
    global p
    
    _device_index = 0
    min_range = 0
    max_range = 100

    user_value = int(frame.text_ctrl.GetValue())
    
    # Check if the input is within the valid range
    # obtain user text input for device selection
    if min_range <= user_value  <= max_range:
        _device_index = user_value
        wx.MessageBox(f"Device {_device_index}: {p.get_device_info_by_index(_device_index)}","Info", wx.OK | wx.ICON_INFORMATION)
    else :
        wx.MessageBox(f"Error: The number must be between {min_range} and {max_range}.","Info", wx.OK | wx.ICON_INFORMATION)


def apply_a_weighting(audio_data):
    """Apply the A-weighting filter to the signal"""
    
    # A-weighting filter design
    # b, a = signal.iirfilter(4, A_WEIGHTING_FREQUENCY, btype='bandpass', ftype='butter', fs=RATE)
    
    # convert to float for filtering
    float_array  = audio_data.astype(np.float32)
    

    # Apply A-weighting
    float_array_filt = A_weight(float_array, RATE)
    
    #convert back to integer for further processing
    int_array   = float_array_filt.astype(np.int16)
    
    print(f'audio_data: {audio_data}')
    print(f'float_array: {float_array}')
    print(f'float_array_filt: {float_array_filt}') 
    print(f'int_array: {int_array}')   
    
    return ( int_array )


def func_calc_SPL():
 
    # input data
    global audio_data
    global a_weighted_signal
    global system_calibration_factor_94db
    
    #output data
    global audio_data_max_pcm_value
    global audio_data_pcm_abs
    global audio_data_mV
    global audio_data_pressurePa
    global audio_data_pressurePa_square
    global audio_data_pressurePa_squareMean
    global audio_data_pressurePa_rms
    global audio_data_pressurePa_rms_calib
    global audio_data_pressurePa_spl

    
    
    # reset output arrays :
    a_weighted_signal = np.zeros(a_weighted_signal.shape)
    audio_data_pcm_abs = np.zeros(audio_data_pcm_abs.shape)
    audio_data_mV = np.zeros(audio_data_mV.shape)
    audio_data_pressurePa = np.zeros(audio_data_pressurePa.shape)
    audio_data_pressurePa_square = np.zeros(audio_data_pressurePa_square.shape)
    audio_data_pressurePa_squareMean = np.zeros(audio_data_pressurePa_squareMean.shape)
    audio_data_pressurePa_rms = np.zeros(audio_data_pressurePa_rms.shape)
    audio_data_pressurePa_rms_calib = np.zeros(audio_data_pressurePa_rms_calib.shape)
    audio_data_pressurePa_spl = np.zeros(audio_data_pressurePa_spl.shape)

    
    # Apply A-weighting to the signal
    # A-weighting does not have an impcat on microphone calibration at 1000Hz, because weighting is 1 at 1000Hz. 
    # Convert to float for A-weighting processing ( scipy.signal requires type 'signal', thus float32)
    a_weighted_signal = apply_a_weighting(audio_data)

    # Check  the maximum absolute value
    audio_data_max_pcm_value_new = np.max(np.abs(a_weighted_signal))
    
    # save if highest of all chunks
    audio_data_max_pcm_value = max(audio_data_max_pcm_value, audio_data_max_pcm_value_new)
    # Example: Process the audio (e.g., calculate RMS for volume level) 
        
    # Absolute values first      
    audio_data_pcm_abs = np.abs(a_weighted_signal)
    
    # Convert to mV using the Sensitivy value of the microphone/ Assuming that the microphone uses the full int16 signale range . 
    # Applying a preamp gain factor (only for better understanding)
    audio_data_mV = audio_data_pcm_abs / PRE_AMP_GAIN_LIN
    print(f"audio_data_mV: {audio_data_mV}")
    
    
    # Converting mV to Pa using the Sensitivy value
    audio_data_pressurePa = audio_data_mV / MIC_SENSITIVITY
    print(f"audio_data_pressurePa: {audio_data_pressurePa}")
    
    # Convert to RMS - Root mean square
    audio_data_pressurePa_square = audio_data_pressurePa ** 2    
    audio_data_pressurePa_squareMean =  np.mean(audio_data_pressurePa_square)         
    if audio_data_pressurePa_squareMean > 0 :
        audio_data_pressurePa_rms = np.sqrt(audio_data_pressurePa_squareMean)
    else :
        audio_data_pressurePa_rms = 0           
    print(f"audio_data_pressurePa_rms: {audio_data_pressurePa_rms}")   
    
    audio_data_pressurePa_rms_calib =audio_data_pressurePa_rms* system_calibration_factor_94db
    
    # convert RMS to SPL (explained below)
    # Reference pressure in air = 20 µPa
    if audio_data_pressurePa_rms_calib > 0:
        audio_data_pressurePa_spl = 20 * np.log10(audio_data_pressurePa_rms_calib / REFERENCE_PRESSURE)    
    else :
        audio_data_pressurePa_spl = 0     
        
    print(f"SPL (dB): {audio_data_pressurePa_spl}")


def func_process_audio_input(frame):
    global p
    global stream
    global frames
    global is_recording
    global _device_index
    
    global max_value
    global audio_data 
    global audio_data_pcm_abs
    global audio_data_mV
    global audio_data_pressurePa
    global audio_data_pressurePa_square
    global audio_data_pressurePa_squareMean
    global audio_data_pressurePa_rms
    global audio_data_pressurePa_rms_calib
    global audio_data_pressurePa_spl
    global system_calibration_factor_94db
    
    global chunk_index_i
    global chunk_noise_list_index
    global chunk_noise_list_spl
    
    print("recording_thread started 1/2")
    
    
    #reset audio input related lists and counters
    chunk_index_i = 0    # counter of processed chunks
    chunk_noise_list_index = []
    chunk_noise_list_spl = []
    frames = []
    
    while is_recording:    
        # Read a chunk of audio data
        data = stream.read(CHUNK)
      
        # Convert the audio data to a numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        frames.append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL()
        
        #chunk counter
        chunk_index_i = chunk_index_i+1
        wx.CallAfter(frame.update_status,  f"Recording running. Number of chunks processed: {chunk_index_i}")
        
        if audio_data_pressurePa_spl > SPL_MAX_DAY_DBA :
            chunk_noise_list_index.append(chunk_index_i)
            chunk_noise_list_spl.append(audio_data_pressurePa_spl)
        
    wx.CallAfter(frame.update_status,  f"Recording terminated. Number of chunks processed: {chunk_index_i}")    


    print("Recording thread stopped.")
    wx.CallAfter(frame.update_status,  "Recording stopped ...")

    print(f"chunk_noise_list_index : {chunk_noise_list_index}")

    
    
def func_run_calibration():
    global p
    global stream
    global frames
    global is_recording
    global  _device_index
    
    global audio_data_max_pcm_value
    global audio_data 
    global audio_data_pcm_abs
    global audio_data_mV
    global audio_data_pressurePa
    global audio_data_pressurePa_square
    global audio_data_pressurePa_squareMean
    global audio_data_pressurePa_rms
    global audio_data_pressurePa_rms_calib
    global audio_data_pressurePa_spl
    global system_calibration_factor_94db


    calib_arr = []
    i= 0 
    audio_data_max_pcm_value  = 0
    
    
    while i<CALIB_ITERATION_LENGTH :
        
        i=i+1
    
        # Read a chunk of audio data
        data = stream.read(CHUNK)
      
        # Convert the audio data to a numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        frames.append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL()
        
        
        if audio_data_pressurePa_rms > 0 : 
            
            # 94 dB != 20 log (rms /p_0) :
            system_calibration_factor_94db_new = (REFERENCE_PRESSURE * (np.power(10, 94/20))) /audio_data_pressurePa_rms
            print(f"system_calibration_factor_94db_new: {system_calibration_factor_94db_new}") 
            
            # Store new value in array 
            calib_arr.append(system_calibration_factor_94db_new)
            
        else :
            print("SPL or Pa equal to zero in this chunk. Check sound input ! ") 
        
         
    # Check if the input PCM coded signal at 94dB calibration db (which is quite loud) is using the full range of the sint16 signal range
    # Check maximum across all chunks, see while loop
    print(f"Maximum PCM  amplitude: {audio_data_max_pcm_value}")
    if audio_data_max_pcm_value > int(MAX_INT16*0.95) :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {audio_data_max_pcm_value}/{MAX_INT16}. Upper threshold: {int(MAX_INT16*0.95)} . Reduce GAIN on PreAmp !","Info", wx.OK | wx.ICON_INFORMATION)       
    elif audio_data_max_pcm_value < int(MAX_INT16*0.8) :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {audio_data_max_pcm_value}/{MAX_INT16}. lower threshold: {int(MAX_INT16*0.8)}  . Increase GAIN on PreAmp !","Info", wx.OK | wx.ICON_INFORMATION)      
    else :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {audio_data_max_pcm_value}/{MAX_INT16}. PreAmp GAIN OK !","Info", wx.OK | wx.ICON_INFORMATION)
    
    # calculate average calibration factor across all chunks, see while loop
    calib_average = sum(calib_arr) / len(calib_arr)
    
    #store average as new calibration factor
    system_calibration_factor_94db = calib_average
    print(f"Averaged system_calibration_factor_94db: {system_calibration_factor_94db}") 
       



def func_check_calibration():
    global p
    global stream
    global frames
    global is_recording
    global  _device_index

    global max_value
    global audio_data 
    global audio_data_pcm_abs
    global audio_data_mV
    global audio_data_pressurePa
    global audio_data_pressurePa_square
    global audio_data_pressurePa_squareMean
    global audio_data_pressurePa_rms
    global audio_data_pressurePa_rms_calib
    global audio_data_pressurePa_spl
    global system_calibration_factor_94db
       
    

    spl_error_arr = []
    spl_error_arr_square = []
    
    i= 0 
   
    while i<CALIB_ITERATION_LENGTH :       
        i=i+1
    
        # Read a chunk of audio data
        data = stream.read(CHUNK)
      
        # Convert the audio data to a numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        frames.append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL()
        
        
        if audio_data_pressurePa_spl > 0 : 
            
            # calculate error in dB SPL
            spl_error = 94 - audio_data_pressurePa_spl
            print(f"spl_error: {spl_error}")
            
            spl_error_square = np.power(spl_error, 2)
                                                
            # Store new value in array 
            spl_error_arr_square.append(spl_error_square)
            spl_error_arr.append(spl_error)
            
        else :
            print("SPL or Pa equal to zero in this chunk. Check sound input ! ") 

                          
    # calculate root mean square error 
    spl_error_average = np.sqrt(np.average(spl_error_square ))
    print(f"spl_error_average: {spl_error_average}")

       

def func_on_button_start_click(frame):
    global is_recording
    global is_logging 
    
    # Enable / Disable buttons
    frame.button_start.Disable()
    frame.button_stop.Enable()
    
    # Start recording thread
    wx.CallAfter(frame.update_status,  "Start button pressed...")
    print(f"is_recording: {is_recording}")
    if not is_recording:
        is_recording = True
        print(f"is_recording: {is_recording}")
        # Create a separate thread to run the process
        # The thread is required to decouple the input stream reading from the GUI app 

        if frame.recording_thread is None or not frame.recording_thread.is_alive():
            print("recording thread will be created now ")
            
            # Argument : frame. Required to create a thread from inside the GUI that serves as longrunning
            # background task. And must refresh the GUI (frame instance) from inside the backround task via callAfter
            frame.recording_thread =threading.Thread(target=func_process_audio_input, args=(frame,))
            print("recording thread created")
            
            frame.recording_thread.start()
            print()
            print("recording thread started 2/2")
            
            # Update the status text after the task is complete (safely in the main thread)
            wx.CallAfter(frame.update_status,  "recording thread started 2...")
        else:
            wx.CallAfter(frame.update_status,  "recording thread is already running.")
            print("recording thread is already running 1.")
    else:
        wx.CallAfter(frame.update_status,  "recording thread is already running.")
        print("recording thread is already running 2.")

    print(f"logging thread will be started: {is_logging}")
    
    # Start logging thread
    if not is_logging:
        is_logging = True
        print(f"is_logging: {is_logging}")
        # Create a separate thread to run the process
        # The thread is required to decouple the input stream reading from the GUI app 

        if frame.logging_thread is None or not frame.logging_thread.is_alive():
            print("logging thread will be created now ")
            
            # Argument : frame. Required to create a thread from inside the GUI that serves as longrunning
            # background task. And must refresh the GUI (frame instance) from inside the backround task via callAfter
            frame.logging_thread =threading.Thread(target=func_saveWave_on_noise_event, args=(frame,))
            print("logging thread created")
            frame.logging_thread.start()
            print("logging thread started 2/2")
            # Update the status text after the task is complete (safely in the main thread)
            wx.CallAfter(frame.update_status,  "logging Thread started 2...")
        else:
            wx.CallAfter(frame.update_status,  "logging Thread is already running.")
            print("logging Thread is already running 1.")
    else:
        wx.CallAfter(frame.update_status,  "logging Thread is already running.")
        print("logging Thread is already running 2.")
        #func_saveWave_on_noise_event()

def func_on_button_stop_click(frame):
    global is_recording
    global is_logging
    
    # Enable / Disable buttons
    frame.button_start.Enable()
    frame.button_stop.Disable()
    
    print(f"is_recording: {is_recording}")
    is_recording = False
    print(f"is_recording: {is_recording}")
    
    print(f"is_logging: {is_logging}")
    is_logging = False
    print(f"is_logging: {is_logging}")
    
    if frame.recording_thread is not None and frame.recording_thread.is_alive():
        frame.recording_thread.join()  # Wait for the thread to finish gracefully
        wx.CallAfter(frame.update_status,  "Recording Thread stopped.")
    else:
        wx.CallAfter(frame.update_status,  "No Recording thread is running.")
    
    if frame.logging_thread is not None and frame.logging_thread.is_alive():
        frame.logging_thread.join()  # Wait for the thread to finish gracefully
        wx.CallAfter(frame.update_status,  "logging Thread stopped.")
    else:
        wx.CallAfter(frame.update_status,  "No logging thread is running.")
   

def func_on_button_runCalib_click(frame):
    
    # Create a separate thread to run the process
    # The thread is required to decouple the input stream reading from the GUI app 
    frame.runCalib_thread   =threading.Thread(target=func_run_calibration)
    frame.runCalib_thread.daemon = True
    frame.runCalib_thread.start()
       
       
def func_on_button_checkCalib_click(frame):
    
    # Create a separate thread to run the process
    # The thread is required to decouple the input stream reading from the GUI app 
    frame.checkCalib_thread  =threading.Thread(target=func_check_calibration)
    frame.checkCalib_thread.daemon = True
    frame.checkCalib_thread.start()


def func_on_button_exit_click(frame):

    # Close the parent application and the GUI event loop (-> indicated by self)
    frame.Close()



def func_saveWave_on_noise_event(frame):
    global p
    global frames
    
    global chunk_index_i
    global chunk_noise_list_index
    global chunk_noise_list_spl
    
    global is_logging
    
    max_spl_in_chunk = 0
    max_spl_index_in_chunk = 0
    max_spl_chunk_index = 0
    start_chunk = 0
    stop_chunk = 0
    
    noise_frames = []
  
    print("logging_thread started 1/2")
    
    while is_logging:  
        
        # Wait for 1 second before running the task again. 
        #Start with offset of 1 sec.
        time.sleep(1)
              
        # check if events are detected and stored in the list
        if len(chunk_noise_list_index) > 0 :
            print()
            print("New Recording event: ")
            print(f"Events detected : {chunk_noise_list_index}")
            print(f"Current chunk processed is : {chunk_index_i}")
        
            #identify max spl value and repsective hunk number 
            max_spl_in_chunk = max(chunk_noise_list_spl)
            max_spl_index_in_chunk = chunk_noise_list_spl.index(max_spl_in_chunk)
            #index of the chunk with maximum spl 
            max_spl_chunk_index = chunk_noise_list_index[max_spl_index_in_chunk]
        
            #extract the relevant noise frames + some delta
            start_chunk = max(max_spl_chunk_index-CHUNK_DNUM, 0)
            stop_chunk = max_spl_chunk_index+CHUNK_DNUM
        
            noise_frames = frames[start_chunk:stop_chunk]
        
            # Get the current local time
            current_time = datetime.datetime.now()
            # Round to the nearest second (remove microseconds)
            rounded_time = current_time.replace(microsecond=0)
        
            # construct file name with relevant data
            noise_file_name = f"{OUTPUT_NOISE_FILENAME}_DataID{max_spl_chunk_index}_dBA{round(max_spl_in_chunk,2)}_Date{rounded_time}.wav"
            # make it compatible with windows filename rules
            noise_file_name = noise_file_name.replace(' ', '_').replace(':', '-')
            print(noise_file_name)

            # Ensure the directory exists (optional, for better handling)
            os.makedirs(OUTPUT_FILE_DIRECTORY, exist_ok=True)

            # Combine directory and file name
            full_path = os.path.join(OUTPUT_FILE_DIRECTORY, noise_file_name)
            print(full_path)
                        
            # Write the recorded data to a WAV file

            with wave.open(full_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(noise_frames))
                print(f"Audio saved as {noise_file_name}")


            #erase noise event arrays
            chunk_noise_list_index = []
            chunk_noise_list_spl = []
            
            max_spl_in_chunk = 0
            max_spl_index_in_chunk = 0
            max_spl_chunk_index = 0
            start_chunk = 0
            stop_chunk = 0
            noise_frames = []
        
            #remove chunks from wave output which are already treated. To free local resources.
            frames = frames[start_chunk:]
            wx.CallAfter(frame.update_status,  f"DataID_{max_spl_chunk_index}__dB_{max_spl_in_chunk}__Horaire:_{rounded_time}.wav")
            print()



def func_on_saveWave_exit_click():
    global p
    global frames
    
    
    # Write the recorded data to a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        print(f"Audio saved as {OUTPUT_FILENAME}")

    # open the recorded data to a WAV file
    with wave.open(OUTPUT_FILENAME, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()  # Sample rate (samples per second)
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
           
        raw_data = wav_file.readframes(num_frames)
 
    
    # read audio data and apply weighting filter    
    # Convert raw byte data into a numpy array
    # For 16-bit audio (common for WAV), use np.int16
    audio_data = np.frombuffer(raw_data, dtype=np.int16)
    # A-weighted audio data
    audio_data_weighted =   apply_a_weighting(audio_data )
    # Create a time axis for plotting
    time = np.linspace(0, num_frames / sample_rate, num_frames)
   
   
   
    # FFT  
    # Perform FFT on the audio signal
    fft_signal = np.fft.fft(audio_data)
    # Perform FFT on the audio signal
    fft_signal_weighted = np.fft.fft(audio_data_weighted)

    # Compute the corresponding frequencies
    frequencies = np.fft.fftfreq(len(fft_signal), d=1/RATE)
    
    # Get the magnitude of the FFT
    fft_magnitude = np.abs(fft_signal)
        # Get the magnitude of the FFT
    fft_magnitude_weighted = np.abs(fft_signal_weighted)
    
    # We only want the positive frequencies
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_magnitude = fft_magnitude[:len(frequencies)//2]
    positive_magnitude_weighted = fft_magnitude_weighted[:len(frequencies)//2]    
 
    '''
    # Plot the waveform
    plt.figure(figsize=(10, 6))
    plt.plot(time, audio_data, color='blue')
    plt.title('Raw audio time domain')
    plt.xlabel("Time [s]")
    plt.ylabel('PCM encoded audio [sint16]')
    plt.grid(True)
    plt.show()        
    '''
    
    # plot frames in time domaine
    # plot process frames in time domaine
    # plot frames FFT
    # plot process frames FFT
    
    # Create some data for plotting
    x1 = time
    y1 = audio_data

    x2 = time
    y2 = audio_data_weighted
    
    x3 = positive_frequencies
    y3 = positive_magnitude
    
    x4 = positive_frequencies
    y4 = positive_magnitude_weighted
    
    

 
    ymin_t = MAX_INT16 = np.iinfo(np.int16).min
    ymax_t = MAX_INT16 = np.iinfo(np.int16).max
    
    
    xmin_f = 0
    xmax_f = 16000
    
    ymin_f = 0
    ymax_f = max(max(positive_magnitude_weighted),max(positive_magnitude))


    
    # Create a 2x2 grid of subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    # First plot (top-left)
    axs[0, 0].plot(x1, y1)
    axs[0, 0].set_title('Raw audio time domain')
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel('PCM encoded audio [sint16]')
    axs[0, 0].set_ylim(ymin_t, ymax_t)
    
    # Third plot (bottom-left)
    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_title('A-weighted audio time domain')
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel('PCM encoded audio [sint16]')
    axs[1, 0].set_ylim(ymin_t, ymax_t)
    
    # Second plot (top-right)
    axs[0, 1].plot(x3, y3)
    axs[0, 1].set_title('Raw audio frequency domain')
    axs[0, 1].set_xlabel("Freq. [kHz]")
    axs[0, 1].set_ylabel('PCM encoded audio [sint16]')
    axs[0, 1].set_xlim(xmin_f, xmax_f)
    axs[0, 1].set_ylim(ymin_f, ymax_f)
    
    # Fourth plot (bottom-right) with a different x-axis range
    axs[1, 1].plot(x4, y4)
    axs[1, 1].set_title('A-weighted audio frequency domain')
    axs[1, 1].set_xlabel("Freq. [kHz]")
    axs[1, 1].set_ylabel('PCM encoded audio [sint16]')
    axs[1, 1].set_xlim(xmin_f, xmax_f)
    axs[1, 1].set_ylim(ymin_f, ymax_f)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.show()
    
    
    
# GUI ############################################"
# Define the main application frame
class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(350, 350))
        
        # Load settings from previous sessions
        global config
        global _device_index
        global system_calibration_factor_94db
        
        print(f"_device_index [default]:  {_device_index}")
        print(f"system_calibration_factor_94db[default]:  {system_calibration_factor_94db}")
        
        # Read Settings
        config.read('config.ini')
        
        # Access values from the config
        _device_index                   = config.getint('Settings', "_device_index")
        system_calibration_factor_94db  = config.getfloat('Settings', "system_calibration_factor_94db") 

        # getint is used for integers
        print(f"_device_index[loaded from config file]:  {_device_index}")
        print(f"system_calibration_factor_94db[loaded from config file]:  {system_calibration_factor_94db}")

        # Print the commit hash
        print("Commit version:", get_commit_version())
        
        # Create a panel inside the frame
        panel = wx.Panel(self)
   
        # To store reference to the thread
        self.recording_thread   = None
        self.logging_thread     = None 
        self.runCalib_thread    = None
        self.checkCalib_thread  = None
              
        # Create a text box for user input
        self.text_ctrl   = wx.TextCtrl(panel, pos=(290, 40), size=(30, 25))

        # Create text output field
        self.status_text = wx.StaticText(panel, label="Status : Start", pos=(10, 230), size=(300, 50))

        # Create a button on the panel
        self.button_checkDevices = wx.Button(panel, label="CheckDevices", pos=(200, 10))

        # Create a button on the panel
        self.button_setDevices = wx.Button(panel, label="SetDevices", pos=(200, 42))

        # Create a button on the panel
        self.button_start = wx.Button(panel, label="Start Measurement!", pos=(10, 10))

        # Create a button on the panel
        self.button_stop = wx.Button(panel, label="Stop Measurement!", pos=(10, 40))
        
        # Create a button on the panel
        self.button_runCalib = wx.Button(panel, label="Calibrate!", pos=(10, 80))
 
        # Create a button on the panel
        self.button_checkCalib = wx.Button(panel, label="Check Calibration!", pos=(10, 110))
        
        # Create a button on the panel
        self.button_exit = wx.Button(panel, label="Close Application", pos=(190, 280))
 
        # Create a button on the panel
        self.button_saveWave = wx.Button(panel, label="Save & Plot Wave File [check signal]", pos=(10, 180))

        
        ##################################################################
        
        
        # Bind the button click event to an event handler function
        self.button_checkDevices.Bind(wx.EVT_BUTTON, self.on_button_checkDevices_click)
 
        # Bind the button click event to an event handler function
        self.button_setDevices.Bind(wx.EVT_BUTTON, self.on_button_setDevices_click)
        
        # Bind the button click event to an event handler function
        self.button_start.Bind(wx.EVT_BUTTON, self.on_button_start_click)
        
        # Bind the button click event to an event handler function
        self.button_stop.Bind(wx.EVT_BUTTON, self.on_button_stop_click)
        
        # Bind the button click event to an event handler function
        self.button_runCalib.Bind(wx.EVT_BUTTON, self.on_button_runCalib_click)
 
        # Bind the button click event to an event handler function
        self.button_checkCalib.Bind(wx.EVT_BUTTON, self.on_button_checkCalib_click)
   
        # Bind the button click event to an event handler function
        self.button_exit.Bind(wx.EVT_BUTTON, self.on_button_exit_click)
   
        # Bind the button click event to an event handler function
        self.button_saveWave.Bind(wx.EVT_BUTTON, self.on_button_saveWave_click)
   
    
        # Show the window
        self.Show()


    def update_status(self, text):
        # Safely append text to the TextCtrl
        self.status_text.SetLabel(text)


    def on_button_checkDevices_click(self, event):
        """Event handler function for the button click."""
        wx.MessageBox("Check Device List in console and set device number in the text box!!", "Info", wx.OK | wx.ICON_INFORMATION)
        # Call the check device function
        func_check_devices()

            
    def on_button_setDevices_click(self, event):  
        # Call function
        func_on_button_setDevices_click(self)


    def on_button_start_click(self, event):
        # Call function
        func_on_button_start_click(self)

            
    def on_button_stop_click(self, event):       
        # Call function
        func_on_button_stop_click(self)


    def on_button_runCalib_click(self, event):
        # Call function
        func_on_button_runCalib_click(self)


    def on_button_checkCalib_click(self, event):
        # Call function
        func_on_button_checkCalib_click(self)
        

    def on_button_exit_click(self, event):
        # Call function
        # need self arguments to know which class instance to close
        func_on_button_exit_click(self)
        
        
    def on_button_saveWave_click(self, event):
        # Call function
        # need self arguments to know which class instance to close
        func_on_saveWave_exit_click()
        
    


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="Task Scheduler GUI")
        return True

# Main GUI application loop ###################################
if __name__ == "__main__":
    app = MyApp()

    # Wrap the main event loop in a try-except block
    try:
        app.MainLoop()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # You can add additional cleanup or logging here if needed
    
    finally:
        # Add settings to the config
        # Add USB device index
        # Add calibration value
        config.set("Settings","_device_index", f"{_device_index}")
        config.set("Settings","system_calibration_factor_94db", f"{system_calibration_factor_94db}") 
        print(f"_device_index [saved]:  {_device_index}")
        print(f"system_calibration_factor_94db[saved]:  {system_calibration_factor_94db}")
    
        # Save the program state (configuration) to a file
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        

    
        # Close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
    
        # Close audio interface
        p.terminate()
        chunk_index_i = 0    # counter of processed chunks
        chunk_noise_list_index = []
        chunk_noise_list_spl = []
        frames = []
        is_recording = False
        is_logging = False
        audio_data                          = np.array([])
        audio_data_pcm_abs                  = np.array([])
        audio_data_mV                       = np.array([])
        audio_data_mV_calib                 = np.array([])
        audio_data_pressurePa               = np.array([])
        audio_data_pressurePa_square        = np.array([])
        audio_data_pressurePa_squareMean    = np.array([])
        audio_data_pressurePa_rms           = np.array([])
        audio_data_pressurePa_rms_calib     = np.array([])
        audio_data_pressurePa_spl           = np.array([])
        audio_data_max_pcm_value            = 0
    
        print("Application has finished.")
  
##############################################################################