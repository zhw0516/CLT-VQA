#### Overcoming Dual Drift for Continual Long-Tailed Visual Question Answering (accepted to ICCV2025)
## Setup
```bash
# Create python environment (optional)
conda create -n cltvqa python=3.7
source activate cltvqa

# Install python dependencies
pip install -r requirements.txt

# Download T5 backbone checkpoint
python download_backbones.py
```

## Code structure
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        featuers/
    karpathy_train_q_{task}.json
    karpathy_val_q_{task}.json
    karpathy_test_q_{task}.json
    trainval_ans2label.json
    trainval_label2ans.json
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json

# Run feature extraction
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5_our.py                                    <= Our VL-T5 model classes
        vqacl.py, vqa_data_memory.py, vqa_model.py ...        <= Testing in the CLT-VQA setting 
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for pretraining and finetuning
```

## CLT-VQA Tasks
```bash
# Training with 1 gpu for VQA v2
cd VL-T5/
bash scripts/VQACL_train.sh 1 # Standard Training

```
