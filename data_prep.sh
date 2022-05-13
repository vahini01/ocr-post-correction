PRED="$HOME/vahini/ainudata/ainu/training/ainu_ocr/"
GOLD="$HOME/vahini/ainudata/ainu/training/ainu_gold/"
PRETRAIN="$HOME/vahini/ainudata/ainu/pretraining/ainu_ocr/"
OUT1="$HOME/way2/ocr-post-correction/sample1/"
OUT2="$HOME/way2/ocr-post-correction/sample2/"
OUT3="$HOME/way2/ocr-post-correction/sample3/"
OUT4="$HOME/way2/ocr-post-correction/sample4/"
OUT5="$HOME/way2/ocr-post-correction/sample5/"

python utils/prepare_data.py --unannotated_src1 $PRETRAIN --annotated_src1 $PRED --annotated_tgt $GOLD --output_folder $OUT1
python utils/prepare_data.py --unannotated_src1 $PRETRAIN --annotated_src1 $PRED --annotated_tgt $GOLD --output_folder $OUT2
python utils/prepare_data.py --unannotated_src1 $PRETRAIN --annotated_src1 $PRED --annotated_tgt $GOLD --output_folder $OUT3
python utils/prepare_data.py --unannotated_src1 $PRETRAIN --annotated_src1 $PRED --annotated_tgt $GOLD --output_folder $OUT4
python utils/prepare_data.py --unannotated_src1 $PRETRAIN --annotated_src1 $PRED --annotated_tgt $GOLD --output_folder $OUT5

