# from https://github.com/NVIDIA/tacotron2
# Modified by Ajinkya Kulkarni

from text import symbols
################################
# Global Parameters#
################################
language = 'French'

################################
# Experiment Parameters#
################################
epochs=500
iters_per_checkpoint=1000
seed=1234
dynamic_loss_scaling=True
fp16_run=True
distributed_run=False
dist_backend="nccl"
dist_url="tcp://localhost:54321"
cudnn_enabled=True
cudnn_benchmark=False
ignore_layers=['embedding.weight']

################################
# Data Parameters     #
################################
load_mel_from_disk=False
training_files='../../../filelists/French_filelist_train.txt'
validation_files='../../../filelists/French_filelist_val.txt'
text_cleaners=['english_cleaners']

################################
# Audio Parameters     #
################################
max_wav_value=32768.0
sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin=0.0
mel_fmax=8000.0

################################
# Model Parameters     #
################################
if language == 'English':
    n_symbols=len(symbols)
else:
    n_symbols = 36 # for French (to support input sequence for other languages)

symbols_embedding_dim=256

# Encoder parameters
encoder_kernel_size=5
encoder_n_convolutions=3
encoder_embedding_dim=256

# Decoder parameters
n_frames_per_step=1  # currently only 1 is supported
decoder_rnn_dim=1024
prenet_dim=256
max_decoder_steps=1000
gate_threshold=0.5
p_attention_dropout=0.1
p_decoder_dropout=0.1

# Attention parameters
attention_rnn_dim=1024
attention_dim=128

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=31

# Mel-post processing network parameters
postnet_embedding_dim=512
postnet_kernel_size=5
postnet_n_convolutions=5

################################
# Optimization Hyperparameters #
################################
use_saved_learning_rate=False
learning_rate=1e-3
weight_decay=1e-6
grad_clip_thresh=1.0
batch_size=1
mask_padding=True  # set model's padded outputs to padded values




#############################################
# Reference Encoder Network Hyperparameters #
#############################################


speaker_encoder_type = 'x-vector'
expressive_encoder_type = 'x-vector'


emotion_classes = 7
speaker_classes = 5


cat_lambda = 0.0
cat_incr = 0.01
cat_step = 1000
cat_step_after = 10
cat_max_step = 300000

kl_lambda = 0.00001
kl_incr = 0.000001
kl_step = 1000
kl_step_after = 500
kl_max_step = 300000



# reference_encoder
ref_enc_filters=[32, 32, 64, 64, 128, 128]
ref_enc_size=[3, 3]
ref_enc_strides=[2, 2]
ref_enc_pad=[1, 1]
ref_enc_gru_size=128

# Style Token Layer
token_num=10
num_heads=8

# embedding size
token_embedding_size=256

# gmvae
num_mixtures = 1

# xvector
input_dim = n_mel_channels
output_dim = token_embedding_size





















