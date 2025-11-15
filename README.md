# CaptiGAN

This project implements a hybrid image captioning system for CIFAR-100 using a DCGAN-based image encoder, a CLIP text encoder, and an LSTM-based decoder. The pipeline supports both simple (class name) and enriched (sentence) captions, and includes experiment management, evaluation, and reporting utilities.

## Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- transformers
- tqdm
- nltk
- rouge
- matplotlib
- pandas
- PIL
- (Optional) optuna for DCGAN hyperparameter search

Install dependencies (if using a virtual environment):
```sh
pip install torch torchvision transformers tqdm nltk rouge matplotlib pandas pillow optuna
```

## Data Preparation
1. Download and extract CIFAR-100:
   - The code will automatically download and extract the dataset to `data/cifar-100-python` if not present.
2. The directory structure should look like:
   ```
   data/
     cifar-100-python/
       train
       test
       meta
   ```

## Training DCGAN
To train a DCGAN on CIFAR-100 images:
```sh
python dcgan_CIFAR100.py
```
- Models and training history will be saved under `experiments/experiment_TIMESTAMP/`.
- For Optuna-based hyperparameter search, see the script for details.

## Training the Captioning Model (LSTM Decoder)
To run a grid search over learning rates and contrastive loss weights for the LSTM captioning model:
```sh
python ques2.py --discriminator hp1 --device cuda --epochs 250 --batch_size 32 --caption_mode simple --decoder_type default
```
- Replace the DCGAN Discriminator options to the `discriminator` parameter and hard code the paths to the generated DCGAN Discriminator models in the `main` block.
- This will create a new experiment directory under `experiments/captioning_grid_search/experiment_TIMESTAMP/`.
- Each hyperparameter combination will be saved in a subfolder (e.g., `lr_5e-05_weight_0.01`).
- Results, models, and logs are saved in each subfolder.
- For enriched captions, use `--caption_mode enriched`.
- For the simple decoder, use `--decoder_type simple`.

## Evaluating Captioning Models
To evaluate a trained LSTM captioning model:
```sh
python eval_LSTM.py --model_path experiments/captioning_grid_search/experiment_TIMESTAMP/lr_XX_weight_XX/captioning_model.pth --decoder_type default --discriminator experiments/experiment_20251112_210616/dcgan_discriminator.pth
```
- Replace `experiment_TIMESTAMP` and `lr_XX_weight_XX` with the desired experiment and hyperparameter folder.
- Use `--decoder_type simple` if the model was trained with the simple decoder.
- Evaluation results (BLEU, ROUGE) and sample images will be saved in the same folder.

## Evaluating DCGAN Results
To plot DCGAN loss curves and summarize Optuna results:
```sh
python eval_DCGAN.py
```
- Edit `EXPERIMENT_DIR` in `eval_DCGAN.py` to point to the desired experiment folder.
- Plots and summaries will be saved in the experiment directory.

## Reproducing Results
1. Run `dcgan_CIFAR100.py` to train the DCGAN (or use provided checkpoints).
2. Run `ques2.py` to train the captioning model with your desired settings.
3. Use `eval_LSTM.py` to evaluate and visualize captioning results.
4. Use `eval_DCGAN.py` to analyze DCGAN training and Optuna search results.

## Experiment Outputs
- All experiment outputs (models, logs, results) are saved under `experiments/`.
- Captioning grid search results are organized as:
  ```
  experiments/captioning_grid_search/experiment_TIMESTAMP/
    lr_XX_weight_XX/
      captioning_model.pth
      train_val_losses.json
      val_caption_results.json
      ...
  ```
- DCGAN results are under `experiments/experiment_TIMESTAMP/`.

## Results

The evaluation results from different trained LSTM image captioning models on CIFAR-100 reveal significant challenges in generating accurate and diverse captions. In one model, the generated captions are highly repetitive, often producing phrases like “a chimpanzee city a city a city...” regardless of the input image, indicating severe mode collapse and a failure to learn meaningful associations between images and captions. In another model, the output is consistently “a wolf” for all images, further highlighting the model’s tendency to default to a single prediction. These patterns suggest that the models struggle with both overfitting and underfitting, likely due to limited caption diversity, insufficient regularization, or suboptimal hyperparameters. The results underscore the difficulty of training LSTM-based captioning systems on small or simple datasets like CIFAR-100, where class names or enriched captions may not provide enough linguistic variety or context for robust learning.

## Next Steps to Improve Model Performance

- **Increase Caption Diversity:** Use more descriptive or enriched captions instead of simple class names to provide richer language targets.
- **Tune Hyperparameters:** Experiment with learning rates, batch sizes, and contrastive loss weights to find more optimal settings.
- **Regularization:** Add dropout to the LSTM decoder and/or use weight decay to reduce overfitting.
- **Model Architecture:** Try deeper or wider LSTM decoders, or experiment with transformer-based decoders for improved sequence modeling.
- **Data Augmentation:** Apply image augmentations to increase the variety of training data and improve generalization.
- **Longer Training:** Increase the number of epochs and patience for early stopping to allow the model more time to learn.
- **Pretrained Embeddings:** Use pretrained word embeddings for the caption tokens if possible.
- **Evaluation:** Visualize generated captions and images to debug systematic errors and better understand model behavior.

---

For further details, see the docstrings in each script and the comments in `ques2_notes.txt`.
