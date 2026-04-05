Here is the essential breakdown of how everything fits together, what to watch out for, and where to dive in when you start modifying it for your thesis!

### 1. Two Separate Architectures
There are two primary architectures included in this codebase, and you must make sure your configs/scripts align with whichever one you are using:

*   **`model/Preliminary.py` (Baseline Model)**
    *   **What it is:** A standard horizontal-vertical transformer (no recurrent blocks).
    *   **Usage:** Used as the proposed baseline for the **Aalto Mobile** dataset. (Tied natively to `utils/train_config.py` and `train.py`).
*   **`model/Model.py` (Full TypeFormer)**
    *   **What it is:** The novel paper architecture featuring **Block Recurrent Attention** (`model/BlockRecurrentTransformer.py`), allowing hidden states to carry over sequence sliding windows.
    *   **Usage:** Used natively for the **KVC Dataset**  (`utils/KVC_config.py` and `KVC_train.py`).

### 2. Data Pipeline & Dimensions
The codebase handles data uniquely, loading everything into memory as a massive Numpy object array.

*   **Structure:** Evaluated as `N_USERS` → `N_SESSIONS` → `N_KEYSTROKES` → `N_FEATURES`.
*   **Aalto Default:** `15` sessions × `50` keystrokes per sequence × `5` features (hold time, inter-press, inter-release, inter-key, keycode). 
*   **Dataloader:** Look closely at `utils/misc.py` (`KeystrokeSessionTriplet`). The dataset object automatically pairs up combinations of **Anchor, Positive, and Negative** batches on-the-fly for Triplet margin learning.

### 3. Configuration Gotchas
Hyperparameters aren't passed via terminal flags. They rely on strictly defined `argparse` files: `train_config.py`, `KVC_config.py`, and `test_config.py`.

> [!WARNING]
> Because variables are mapped dynamically as object properties (e.g. `configs.hlayers`), if a variable is missing, it skips python syntax checks and crashes at runtime! Always verify that variables map 1:1 if you copy code between dataset scopes.

### 4. How the Flow Works
*   **Training (`train.py`):** Runs blindly for 1,000 epochs. However, it tracks `eer_v` (Validation Equal Error Rate) at the end of every epoch. If it hits a new lowest metric, it silently overwrites your `.pt` model weights file. This means training is gracefully safe to kill `Ctrl+C` once the validation score plateaus.
*   **Testing (`test.py`):** Testing skips standard BCE loss methods and instead generates dense embeddings mapping for every user. It loops a strictly pairwise Euclidean calculation to generate the final global Mean Per-Subject EER %.

### 5. Checkpoints Before Modifying
Before you start experimenting with new ideas (like testing new feature structures or loss formats), knowing where to look is half the battle:

1. **Adding new Data Features:** If you generate 9 features instead of 5 in `preprocess_Aalto.py`, you only need to change `configs.dimensionality = 9` in `train_config.py`. The linear embedding dimensions will organically adapt inside the transformer class—no dense layer rewrites required!
2. **Replacing Triplet Loss:** Triplet Margin loss is calculated explicitly in `utils/misc.py/TripletLoss`. If you want to experiment with ArcFace or SupCon metrics, you will need to map your new objective loss function natively into `train_one_epoch()` directly in `train.py`.
3. **Handling Sequence Sizes:** TypeFormer explicitly expects a strict sliding window layout (e.g., rigid blocks of exactly 50 keystrokes). If you want to feed it raw, variably sized sentences, you will need to do serious masking modifications—otherwise, the recurrent attention gates will wildly misalign their internal dimensions.