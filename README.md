# Gender Classification Web App

A Flask web application that takes an uploaded image, detects faces using a Haar cascade classifier, and predicts the gender of each detected face using a PCA + SVM pipeline. The app is deployment-ready with a Procfile and Aptfile configured for Heroku.

---

## Motivation

Binary gender classification from facial images is a constrained, well-defined problem that makes it a useful test case for classical machine learning on image data. The more interesting engineering problem here is the full pipeline: detection, preprocessing, dimensionality reduction, and classification running together inside a web server that handles image upload, inference, and result rendering in a single request cycle. This project demonstrates that an ML inference pipeline can be packaged as a deployable web application without a deep learning framework — the entire model stack is scikit-learn and OpenCV, which makes it lightweight enough to run on a free-tier cloud host.

---

## How It Works

The inference pipeline runs in ten steps for each detected face:

1. **Face detection:** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) identifies face bounding boxes in the uploaded image
2. **Crop and convert:** Each detected region is cropped and converted to grayscale
3. **Normalize:** Pixel values scaled to 0–1
4. **Resize:** Cropped face resized to 100×100 pixels (INTER_AREA if downscaling, INTER_CUBIC if upscaling)
5. **Flatten:** Reshaped to a 1×10,000 vector
6. **Mean subtraction:** The training mean face vector (stored in `pca_dict.pickle`) is subtracted
7. **PCA transform:** The mean-subtracted vector is projected into 50 principal components (eigenfaces)
8. **SVM prediction:** The 50-dim eigenface vector is passed to the trained SVM classifier
9. **Probability score:** `predict_proba` returns a Platt-scaled confidence score displayed alongside the label
10. **Annotated output:** Bounding boxes and labels drawn on the original image and saved for display

**Why PCA(50, whiten=True) + SVM — not a CNN:**

The training dataset contains ~4,300 face images. A CNN with sufficient capacity to learn gender-discriminative features from scratch would overfit severely at this scale. PCA(50) captures ~80% of cumulative explained variance across the face distribution — the elbow in the explained variance curve falls at ~50 components, after which marginal gain per component drops sharply. `whiten=True` divides each component by its standard deviation before passing to the SVM, so the RBF kernel treats all 50 eigenface dimensions equally rather than being dominated by the first few high-variance components (which encode global illumination, not gender).

**The Haar cascade is not just preprocessing — it is part of the model:**

The training dataset was produced by running the same Haar cascade on raw images to crop face ROIs (training notebook 01). This means the SVM was trained on Haar-detected, Haar-cropped face patches — with the same aspect ratios, scale selection, and boundary artefacts that inference will produce. If the face detector is swapped at inference (e.g., replaced with MTCNN or MediaPipe), the crop boundaries and included background change, the input distribution shifts, and the SVM's accuracy degrades with no error signal. The detector and the classifier form a single contract.

**Mean subtraction design:**

The training mean face is stored separately in `pca_dict.pickle` and subtracted manually at inference before calling `pca.transform()`. sklearn's `PCA.transform()` also subtracts `pca.mean_` internally — but since the PCA was fitted on already mean-subtracted data, `pca.mean_` is approximately zero. The manual subtraction is the actual centering step; sklearn's internal subtraction is a no-op. This pattern is intentional: it avoids relying on sklearn's internal mean across serialisation/deserialisation, which can behave unexpectedly across library versions.

The web layer is a minimal Flask app with three routes: a landing page, an app entry page, and the gender prediction endpoint (GET/POST). The prediction endpoint saves the uploaded image, runs the pipeline, and renders the original face crop, its eigenface reconstruction, the predicted label, and the confidence score.

---

## Project Structure

```
GenderClassifyWA/
├── main.py # Flask app entry point, URL routing
├── requirements.txt # Python dependencies
├── Procfile # Heroku deployment config (gunicorn)
├── Aptfile # System-level dependencies for Heroku (libsm6, libxrender1, etc.)
├── app/
│ ├── face_recognition.py # Full inference pipeline: detection → PCA → SVM → annotated output
│ └── views.py # Flask view functions: index, app, gender prediction handler
├── model/
│ ├── haarcascade_frontalface_default.xml # OpenCV Haar cascade for face detection
│ ├── model_svm.pickle # Trained SVM classifier (GridSearchCV best estimator)
│ └── pca_dict.pickle # PCA(50, whiten=True) + training mean face vector
├── training_evidence/ # Training notebooks (data not redistributed; notebooks included)
│ ├── 01_FRM_data_preprocessing_crop_faces.ipynb # Haar cascade on raw images → face ROI crops
│ ├── 02_FRM_data_preprocessing_EDA.ipynb # Class distribution, dimension analysis, resolution confound
│ ├── 03_FRM_feature_extraction_eigen_face.ipynb # PCA(50, whiten=True), eigenface visualisation, elbow analysis
│ ├── 04_FRM_Machine_learning.ipynb # GridSearchCV, classification report, Cohen's Kappa
│ ├── 05_FRM_make_pipeline.ipynb # End-to-end pipeline assembly and pickle serialisation
│ ├── 06_FRM_predictions.ipynb # Inference validation on sample images
│ ├── 01_OpenCV_values_or_pixels.ipynb # OpenCV fundamentals reference
│ ├── 02_OpenCV_reading_image.ipynb # OpenCV image I/O reference
│ ├── 03_OpenCV_image_resizing.ipynb # Resize interpolation methods reference
│ └── 04_OpenCV_face_detection.ipynb # Haar cascade parameter tuning reference
├── static/
│ ├── upload/ # Uploaded images saved here
│ └── predict/ # Annotated output images, ROI crops, eigenface images
├── templates/
│ ├── index.html # Landing page
│ ├── app.html # App entry page
│ ├── gender.html # Results page
│ └── base.html # Base template
└── test_images/ # Sample images for manual testing
```

---

## Setup and Installation

**Python version:** 3.8+

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run locally:**
```bash
python main.py
```

The app will be available at `http://127.0.0.1:5000`. Navigate to `/app/gender/` to upload an image and run prediction.

**Important:** The app must be run from the project root directory. The model loading paths in `face_recognition.py` are relative to the project root (`./model/...`). Running from a different working directory will cause a file-not-found error on startup.

**Deploy to Heroku:**
```bash
heroku create
git push heroku main
```
The `Procfile` and `Aptfile` are pre-configured. The `Aptfile` installs the system libraries required by OpenCV headless (`libsm6`, `libxrender1`, `libfontconfig1`, `libice6`).

---

## Results

The app produces per-face output for each uploaded image:
- Predicted label: Male or Female
- Confidence score (SVM probability, displayed as a percentage)
- Annotated image with coloured bounding boxes (yellow for Male, pink for Female)
- Grayscale face crop and eigenface reconstruction displayed alongside the prediction

**Training metrics (from `training_evidence/04_FRM_Machine_learning.ipynb`):**

- **Dataset:** ~4,300 face images, near-balanced male/female split, 80/20 stratified train/test split
- **GridSearchCV:** 3-fold CV over `C` ∈ [0.5–50], `kernel` ∈ [rbf, poly], `gamma` ∈ [0.001–0.1], `coef0` ∈ [0, 1]
- **Classification report:** precision, recall, F1 per class — female recall is higher than male recall, consistent with the resolution bias in the training data (female images are higher-resolution, producing better eigenspace representations)
- **Cohen's Kappa:** 0.7–0.9 range — with near-balanced classes, Kappa ≈ 2×accuracy − 1, placing accuracy in the ~85–90% range
- **Reported AUC: treat with caution.** The training notebook computes `roc_auc_score(np.where(y_test=="male",1,0), np.where(y_pred=="male",1,0))` — passing hard binary predictions instead of probability scores. This computes balanced accuracy, not a proper AUC. The correct call would use `model_final.predict_proba(x_test)[:,1]` as the score argument.

---

## Limitations

- **Resolution confound in training data.** EDA notebook documents that female images are predominantly high-resolution (INTER_AREA downscaling) while male images are more often low-resolution (INTER_CUBIC upscaling). Different interpolation methods produce systematically different pixel textures. The SVM may partially learn interpolation artefacts rather than genuine gender features — inflating test accuracy beyond what represents real discrimination.
- **Multi-face mislabelling in training crops.** The crop loop in notebook 01 iterates over all Haar detections for an image but uses a fixed filename per image, so only the last detected face is saved. A group photo labelled `female` that contains a male face as the last detection produces a mislabelled training example. There is no `len(faces_list) == 1` guard.
- **AUC metric is computed incorrectly.** Hard binary predictions are passed to `roc_auc_score` instead of probability scores. The reported number equals balanced accuracy, not AUC. This is noted here so the number is not quoted out of context.
- **Haar cascade `minNeighbors` mismatch between training and inference.** Training crop pipeline uses `detectMultiScale(gray, 1.5, 5)`; inference uses `detectMultiScale(gray, 1.5, 3)`. Lower `minNeighbors` at inference increases false-positive detections — background patches get classified without error.
- **No reconstruction error quality gate.** The eigenface reconstruction is displayed visually but not used as a confidence signal. A non-face patch (false Haar detection) will have high reconstruction error but the SVM still outputs a confident label.
- **Binary gender classification.** The model outputs only Male or Female, which does not reflect the full range of gender expression.

---

## What I Would Do Differently

**Fix the multi-face contamination in training.** Adding `if len(faces_list) == 1:` before the `cv2.imwrite` call in notebook 01 eliminates mislabelled examples from group photos in a single line. The scale of this contamination is unknown but non-zero.

**Fix the AUC computation.** Replacing `np.where(y_pred=="male",1,0)` with `model_final.predict_proba(x_test)[:,1]` in the `roc_auc_score` call produces a proper ROC curve. The current number is balanced accuracy under a different name.

**Equalise the resolution distribution between classes.** The female/male resolution imbalance is the most structurally damaging issue in the dataset. Either balance it by sub-sampling or ensure both classes go through the same interpolation path. Without this, it is impossible to know how much of the accuracy reflects genuine gender discrimination versus interpolation texture differences.

**Add a reconstruction error quality gate at inference.** `eig_img` (the eigenface reconstruction) is already computed in `face_recognition.py` for display. Computing `||roi_reshape - eig_img||²` and comparing it to the 95th percentile of training reconstruction errors would flag low-quality or non-face inputs as unreliable before the SVM label is shown to the user.

**Fix the concurrent request filename collision.** `pred_filename = 'prediction_image.jpg'` is hardcoded — simultaneous uploads overwrite each other. Replacing it with `str(uuid.uuid4()) + '.jpg'` per request is a one-line fix that makes the app safe for any number of concurrent users.

---

## References

- **Eigenfaces / PCA approach:** Turk, M. & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1), 71–86
- **OpenCV Haar Cascades:** Viola, P. & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. CVPR
- **scikit-learn SVM:** Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830
- **Flask:** Pallets Projects — https://flask.palletsprojects.com/