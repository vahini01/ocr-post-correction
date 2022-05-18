# --------------------- REQUIRED: Modify for each dataset and/or experiment ---------------------

SAMPLE="sample1"
HOME="/home/vahini"
OCRPATH="$HOME/ocr-post-correction/$SAMPLE"
POST_PATH="$HOME/ocr-post-correction"

# Set test source file (test_tgt is optional, can be used to compute CER and WER of the predicted output)
test_src="$OCRPATH/training/test8_src1.txt"
test_tgt="$OCRPATH/training/test8_tgt.txt"

# Set experiment parameters
expt_folder="$OCRPATH/singlesource_10fold_8_seed1_base/"

dynet_mem=1000 # Memory in MB available for testing

params="--pretrain_dec --pretrain_s2s --pretrain_enc"
trained_model_name="my_trained_model"

# ------------------------------END: Required experimental settings------------------------------


# Load the trained model and get the predicted output on the test set (add --dynet-gpu for using GPU)
python $POST_PATH/postcorrection/multisource_wrapper.py \
--dynet-mem $dynet_mem \
--dynet-autobatch 1 \
--test_src1 $test_src \
--test_tgt $test_tgt \
$params \
--single \
--vocab_folder $expt_folder/vocab \
--output_folder $expt_folder \
--load_model $expt_folder"/models/"$trained_model_name \
--testing \
--dynet-gpu


