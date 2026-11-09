# Demo images

Ready-to-use photos for a live demo of `app.py`. **None of them were used for training:**

- `lab/`: PlantVillage leaves from our **held-out test split** (leaf-grouped, never seen in training).
- `field/`: real-world photos from the **PlantDoc test set** (Singh et al., 2020), which is used only for evaluation.

Each image was checked with the trained model in `app_model/`. The expected top prediction is:

### Lab images

| File | Expected prediction | Confidence |
|---|---|---|
| `lab/01_potato_late_blight.jpg` | Potato: Late blight | 96.9% |
| `lab/02_tomato_healthy.jpg` | Tomato: healthy | 96.6% |
| `lab/03_tomato_early_blight.jpg` | Tomato: Early blight | 94.4% |
| `lab/04_apple_apple_scab.jpg` | Apple: Apple scab | 93.3% |
| `lab/05_grape_black_rot.jpg` | Grape: Black rot | 92.0% |
| `lab/06_corn_common_rust.jpg` | Corn (maize): Common rust | 91.4% |

### Field photos

| File | Expected prediction | Confidence |
|---|---|---|
| `field/01_apple_apple_scab.jpg` | Apple: Apple scab | 94.4% |
| `field/02_apple_cedar_apple_rust.jpg` | Apple: Cedar apple rust | 97.9% |
| `field/03_apple_cedar_apple_rust.jpg` | Apple: Cedar apple rust | 99.8% |
| `field/04_corn_northern_leaf_blight.jpg` | Corn (maize): Northern Leaf Blight | 93.2% |
| `field/05_corn_common_rust.jpg` | Corn (maize): Common rust | 96.4% |
| `field/06_squash_powdery_mildew.jpg` | Squash: Powdery mildew | 90.4% |

These field photos were chosen because the model gets them right. On the full PlantDoc test set the model is correct on **57.2%** of photos, so a random field photo may be misclassified.

**Suggested demo order:** `lab/03_tomato_early_blight.jpg` (Grad-CAM lands on the brown spots), then `field/05_corn_common_rust.jpg` (cluttered field photo), then `field/03_apple_cedar_apple_rust.jpg`.

*Image sources: PlantVillage via the New Plant Diseases dataset (Kaggle), and PlantDoc (github.com/pratikkayal/PlantDoc-Dataset). Copyright of the original photos remains with their owners; they are included here for academic demonstration only.*
