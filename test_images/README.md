# Test images (one folder per disease)

195 photos in **38 folders, one per class the model knows**. The folder name is the correct answer. Upload any photo in the app and compare the prediction with the folder name.

**None of these photos were used to train the model.** They were picked **at random**, not because the model gets them right, so they show its real behaviour.

| File name | What it is | Source |
|---|---|---|
| `lab_1.jpg` … `lab_3.jpg` | Single leaf on a plain lab background | Held-out test split (never trained on). For Orange, Soybean and Tomato Yellow Leaf Curl Virus: PlantVillage photos that are not in the training dataset at all |
| `field_1.jpg` … `field_3.jpg` | Real-world photo: clutter, shadows, several leaves | PlantDoc test set (never trained on); only for the 27 classes PlantDoc covers |

## What to expect (measured with the model in `app_model/`)

| Photo type | Correct | Accuracy |
|---|---|---|
| Lab photos | 113 / 114 | **99.1%** |
| Field photos | 42 / 81 | **51.9%** |
| All | 155 / 195 | 79.5% |

This matches the main results (99.34% lab test, 57.20% on the full PlantDoc test set). Lab-style photos are almost always right. Field photos are right about half the time, which is the lab-to-field gap (Gap 3) that more field training data would close.

The full answer key, with the model's prediction and confidence for every photo, is in [`answer_key.csv`](answer_key.csv).

## Patterns in the mistakes (field photos)

- **Similar-looking diseases get confused:** corn northern leaf blight vs gray leaf spot, tomato bacterial spot vs Septoria leaf spot, potato early vs late blight.
- **Right disease, wrong crop:** potato late blight predicted as tomato late blight, pepper bacterial spot as tomato bacterial spot. These crops share the same pathogens.
- **Healthy leaves of one crop called another crop's healthy leaf:** cherry and blueberry leaves predicted as apple.
- **Most wrong answers have low confidence (often below 50%),** so a low score is a useful warning sign.

## Photos of things the model does not know

The model only knows these 38 classes. A photo of another crop (e.g. mango, rice) or of a non-plant object will still get one of the 38 labels, because it has no "unknown" option (Gap 7, future work). On 600 non-plant photos the average confidence was only about 29%, compared with about 84% on real leaves.

*Image sources: PlantVillage (via the New Plant Diseases dataset) and PlantDoc (github.com/pratikkayal/PlantDoc-Dataset). Copyright of the original photos remains with their owners; they are included here for academic testing only.*
