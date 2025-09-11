# dementia-care
Code for the paper 'Affordable audio hardware and artificial intelligence can transform the dementia care pipeline'

One needs to install the libraries of each script

VAD.py: Voice activity detection. Outputs a wav file and a plot

ESR: Emotion Speech Recognition of a folder of recordings. Recs should 16kHz. It is based on uperb/hubert-large-superb-er

diarization, diarization_v2. Expects a trining folder with subfolders with names A, B, C, D, E, F, A+B, A+C. A to F are the speakers. Each single letter folder has recordings from a single speaker. The name of the folder denotes the speaker. A+B means that folder contains recordings each have a conversation with speaker A and B only. It expects an output folder with recordings.

eval_diarization: It takes a single csv as input and outputs evaluation metrics
