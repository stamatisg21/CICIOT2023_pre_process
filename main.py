# ============================================================
# CELL 1 — SETUP & DATA PREPARATION
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, time, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("="*55)
print("  ENVIRONMENT CHECK")
print("="*55)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPUs found : {len(tf.config.list_physical_devices('GPU'))}")
print("="*55)

# ── LABEL MAPPING ────────────────────────────────────────────
attack_group = {
    'DDoS-ICMP_Flood':'DDoS','DDoS-UDP_Flood':'DDoS',
    'DDoS-TCP_Flood':'DDoS','DDoS-PSHACK_Flood':'DDoS',
    'DDoS-SYN_Flood':'DDoS','DDoS-RSTFINFlood':'DDoS',
    'DDoS-SynonymousIP_Flood':'DDoS','DDoS-HTTP_Flood':'DDoS',
    'DDoS-ICMP_Fragmentation':'DDoS','DDoS-UDP_Fragmentation':'DDoS',
    'DDoS-ACK_Fragmentation':'DDoS','DDoS-SlowLoris':'DDoS',
    'DoS-UDP_Flood':'DoS','DoS-SYN_Flood':'DoS',
    'DoS-TCP_Flood':'DoS','DoS-HTTP_Flood':'DoS',
    'Mirai-greeth_flood':'Mirai','Mirai-greip_flood':'Mirai',
    'Mirai-udpplain':'Mirai',
    'Recon-HostDiscovery':'Recon','Recon-OSScan':'Recon',
    'Recon-PortScan':'Recon','Recon-PingSweep':'Recon',
    'VulnerabilityScan':'Recon',
    'DNS_Spoofing':'Spoofing','MITM-ArpSpoofing':'Spoofing',
    'BenignTraffic':'Benign',
    'BrowserHijacking':'Web','Backdoor_Malware':'Web',
    'XSS':'Web','SqlInjection':'Web',
    'Uploading_Attack':'Web','CommandInjection':'Web',
    'DictionaryBruteForce':'BruteForce'
}

BASE_PATH  = './'
TRAIN_PATH = f'{BASE_PATH}/train/train.csv'
TEST_PATH  = f'{BASE_PATH}/test/test.csv'
VAL_PATH   = f'{BASE_PATH}/validation/validation.csv'

def load_and_map(path, name):
    print(f"\nLoading {name}...")
    df = pd.read_csv(path)
    print(f"  Raw shape      : {df.shape}")
    df['label_category'] = df['label'].map(attack_group)
    df = df.drop(columns=['label'])
    df = df.dropna(subset=['label_category'])
    print(f"  Mapped shape   : {df.shape}")
    for cls, cnt in df['label_category'].value_counts().items():
        print(f"    {cls:<15} {cnt:>10,}")
    return df

df_train = load_and_map(TRAIN_PATH, 'TRAIN')
df_test  = load_and_map(TEST_PATH,  'TEST')
df_val   = load_and_map(VAL_PATH,   'VALIDATION')

def clean(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

df_train = clean(df_train)
df_test  = clean(df_test)
df_val   = clean(df_val)
print("\nCleaning done.")

TARGET = 20_000
parts  = []
for lbl in df_train['label_category'].unique():
    chunk = df_train[df_train['label_category'] == lbl]
    parts.append(resample(chunk, replace=len(chunk) < TARGET,
                          n_samples=TARGET, random_state=42))
df_train_bal = (pd.concat(parts)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))
del df_train

le = LabelEncoder()
le.fit(df_train_bal['label_category'])
CLASS_NAMES = list(le.classes_)

y_train = le.transform(df_train_bal['label_category'])
y_test  = le.transform(df_test['label_category'])
y_val   = le.transform(df_val['label_category'])

# ── CAPTURE FEATURE NAMES BEFORE SCALING ─────────────────────
# scaler.fit_transform() returns a plain numpy array — all column
# names are lost at that point.  Grab them NOW while we still have
# the DataFrame, and persist them so every downstream cell can use
# the real names without having to re-read the raw CSV.
X_train_df = df_train_bal.drop(columns=['label_category'])
X_test_df  = df_test.drop(columns=['label_category'])
X_val_df   = df_val.drop(columns=['label_category'])

FEATURE_NAMES = list(X_train_df.columns)   # ← the fix lives here
print(f"\nFeature names captured: {len(FEATURE_NAMES)} features")
print(f"First 5 : {FEATURE_NAMES[:5]}")

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_df)   # numpy arrays from here on
X_test  = scaler.transform(X_test_df)
X_val   = scaler.transform(X_val_df)

n_feat    = X_train.shape[1]
n_classes = len(CLASS_NAMES)

# ── SAVE CSVs WITH REAL COLUMN NAMES ─────────────────────────
# Reconstruct DataFrames so the CSVs have proper headers.
def make_output_df(X, y):
    df_out = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_out['label_encoded']  = y
    df_out['label_category'] = le.inverse_transform(y)
    return df_out

make_output_df(X_train, y_train).to_csv('train_ready.csv', index=False)
make_output_df(X_test,  y_test ).to_csv('test_ready.csv',  index=False)
make_output_df(X_val,   y_val  ).to_csv('val_ready.csv',   index=False)

joblib.dump(scaler,       'scaler.pkl')
joblib.dump(le,           'label_encoder.pkl')
joblib.dump(FEATURE_NAMES,'feature_names.pkl')   # ← persisted for SHAP cell

print("\nSaved: train_ready.csv | test_ready.csv | val_ready.csv")
print("Saved: scaler.pkl | label_encoder.pkl | feature_names.pkl")

# ── CLASS DISTRIBUTION PLOT ───────────────────────────────────
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(12, 5))
colors = plt.cm.Set2(np.linspace(0, 1, len(CLASS_NAMES)))
bars = plt.bar([CLASS_NAMES[i] for i in unique], counts,
               color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f'{val:,}', ha='center', fontsize=10, fontweight='bold')
plt.title('Training Set — Class Distribution (After Balancing)',
          fontsize=14, fontweight='bold')
plt.xlabel('Attack Class')
plt.ylabel('Number of Samples')
plt.ylim(0, TARGET * 1.2)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*55)
print("CELL 1 COMPLETE")
print("="*55)


# ============================================================
# CELL 2 — SIMPLE DNN + SHAP EXPLAINABILITY
# ============================================================
import shap

# ── RELOAD ARTIFACTS if running this cell standalone ─────────
# (safe to run even if Cell 1 already populated these variables)
try:
    FEATURE_NAMES
except NameError:
    FEATURE_NAMES = joblib.load('feature_names.pkl')
    le            = joblib.load('label_encoder.pkl')
    CLASS_NAMES   = list(le.classes_)
    # reload data from saved CSVs
    _tr = pd.read_csv('train_ready.csv')
    _te = pd.read_csv('test_ready.csv')
    _va = pd.read_csv('val_ready.csv')
    X_train = _tr[FEATURE_NAMES].values
    y_train = _tr['label_encoded'].values
    X_test  = _te[FEATURE_NAMES].values
    y_test  = _te['label_encoded'].values
    X_val   = _va[FEATURE_NAMES].values
    y_val   = _va['label_encoded'].values
    n_feat    = X_train.shape[1]
    n_classes = len(CLASS_NAMES)

# ── BUILD DNN ────────────────────────────────────────────────
model = Sequential([
    Input(shape=(n_feat,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=256,
                    callbacks=callbacks, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history['loss'],     label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(history.history['accuracy'],     label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title('Accuracy'); axes[1].legend()
plt.tight_layout()
plt.savefig('dnn_training_curves.png', dpi=300)
plt.show()


# ============================================================
# CELL 3 — RANDOM FOREST + SHAP
# ============================================================
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=20,
                                  n_jobs=-1, random_state=42, verbose=1)
rf_model.fit(X_train, y_train)
print(f"\nRandom Forest Test Accuracy: {rf_model.score(X_test, y_test):.4f}")
joblib.dump(rf_model, 'random_forest_for_shap.pkl')

# ── SHAP ─────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(rf_model)
test_sample = X_test[:500]
shap_raw    = explainer.shap_values(test_sample)

# Normalise to a consistent 3D array (n_samples, n_features, n_classes)
# regardless of SHAP version:
#   old SHAP → list of n_classes arrays, each (n_samples, n_features)
#   new SHAP → single array (n_samples, n_features, n_classes)
if isinstance(shap_raw, list):
    shap_3d = np.stack(shap_raw, axis=-1)   # → (n_samples, n_features, n_classes)
else:
    shap_3d = shap_raw                       # already (n_samples, n_features, n_classes)

print(f"shap_3d shape : {shap_3d.shape}")   # should be (500, n_feat, n_classes)

# mean |SHAP| across samples AND classes → one value per feature
mean_abs_shap = np.mean(np.abs(shap_3d), axis=(0, 2))   # (n_features,)
print(f"mean_abs_shap : {mean_abs_shap.shape} | FEATURE_NAMES : {len(FEATURE_NAMES)}")

# ── PLOT 1: Global top-20 bar ────────────────────────────────
feat_imp = (pd.DataFrame({'feature': FEATURE_NAMES,
                          'mean_abs_shap': mean_abs_shap})
            .sort_values('mean_abs_shap', ascending=False)
            .head(20))

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp, y='feature', x='mean_abs_shap', palette='viridis')
plt.title('Top 20 Features by Mean |SHAP| — Random Forest', fontweight='bold')
plt.xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('shap_top20_features.png', dpi=300)
plt.show()

# ── PLOT 2: Local waterfall ──────────────────────────────────
pred_class = rf_model.predict(test_sample[0:1])[0]
true_class = y_test[0]

expl = shap.Explanation(
    values        = shap_3d[0, :, pred_class],          # (n_features,) for predicted class
    base_values   = explainer.expected_value[pred_class],
    data          = test_sample[0],
    feature_names = FEATURE_NAMES
)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(expl, show=False, max_display=15)
plt.title(f'Local explanation  |  '
          f'True: {le.classes_[true_class]}  '
          f'Pred: {le.classes_[pred_class]}')
plt.tight_layout()
plt.savefig('shap_local_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()

# ── PLOT 3: Summary beeswarm ─────────────────────────────────
# summary_plot also needs the list format → convert back
shap_list = [shap_3d[:, :, c] for c in range(n_classes)]

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_list, test_sample,
                  feature_names=FEATURE_NAMES,
                   class_names=CLASS_NAMES,
                  show=False, max_display=20)
plt.title('SHAP Summary Plot (all classes)', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSHAP analysis complete.")
print("Saved: shap_top20_features.png | shap_local_waterfall.png | shap_summary.png")