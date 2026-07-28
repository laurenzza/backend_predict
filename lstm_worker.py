# =============================================================================
# lstm_worker.py
# Worker subprocess untuk training LSTM PNYB
# =============================================================================
import os
import sys

os.environ['PYTHONHASHSEED']         = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import random
import pandas as pd
import joblib
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
try:
    tf.config.experimental.enable_op_determinism()
    print("[WORKER] enable_op_determinism() aktif.", flush=True)
except AttributeError:
    print("[WORKER][WARNING] enable_op_determinism() tidak tersedia.", flush=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

RANDOM_SEED    = 42
SEQ_LEN        = 6
N_CLUSTER      = 5
BULAN_EKSKLUDE = ['2024-06', '2024-07']
FEATURES = [
    'lag1_r', 'lag2_r', 'lag3_r', 'lag6_r', 'lag12_r',
    'roll3_r', 'roll6_r', 'tren_3m', 'bulan', 'produk_id', 'faktor_libur'
]
KALENDER_LIBUR = {
    '2020-05': 0.40, '2021-05': 0.45, '2022-05': 0.55,
    '2023-04': 0.50, '2024-04': 0.65, '2025-03': 0.90,
    '2020-06': 0.60, '2021-06': 0.60, '2022-06': 0.65,
    '2020-07': 0.60, '2021-07': 0.65, '2022-07': 0.70,
    '2023-06': 0.65, '2024-05': 0.60,
    '2024-06': 0.12, '2024-07': 0.28,
    '2024-08': 0.55, '2024-12': 0.40,
    '2025-01': 0.58,
}

def set_seed_ulang(offset=0):
    random.seed(RANDOM_SEED + offset)
    np.random.seed(RANDOM_SEED + offset)
    tf.random.set_seed(RANDOM_SEED + offset)

def main():
    if len(sys.argv) != 3:
        print("[WORKER][ERROR] Usage: lstm_worker.py <input.pkl> <output_prefix>", flush=True)
        sys.exit(1)

    input_path    = sys.argv[1]
    output_prefix = sys.argv[2]

    print(f"[WORKER] Memulai training LSTM dari: {input_path}", flush=True)
    set_seed_ulang(0)

    data              = joblib.load(input_path)
    mtr_clean         = data['mtr_clean']
    median_qty_produk = data['median_qty_produk']
    produk_cluster    = data['produk_cluster']
    le_classes        = data['le_classes']

    le = LabelEncoder()
    le.classes_ = np.array(le_classes)

    cluster_models, cluster_scalers = {}, {}

    for cid in range(N_CLUSTER):
        df_cl = mtr_clean[mtr_clean['cluster_id'] == cid].reset_index(drop=True)
        if len(df_cl) < 40:
            print(f"[WORKER]   Klaster {cid} dilewati: hanya {len(df_cl)} sampel (<40)", flush=True)
            continue

        set_seed_ulang(cid)

        scaler = MinMaxScaler()
        X_sc   = scaler.fit_transform(df_cl[FEATURES].values.astype(float))
        y_all  = df_cl['log_rasio'].values

        X_seq, y_seq = [], []
        for _, grp in df_cl.groupby('Nama Produk', sort=True):
            local_idx = list(grp.sort_values('bulan_period').index)
            for i in range(len(local_idx) - SEQ_LEN):
                X_seq.append(X_sc[local_idx[i:i + SEQ_LEN]])
                y_seq.append(y_all[local_idx[i + SEQ_LEN]])

        if len(X_seq) < 20:
            print(f"[WORKER]   Klaster {cid} dilewati: sekuens < 20", flush=True)
            continue

        X_3d, y_arr = np.array(X_seq), np.array(y_seq)

        model = Sequential([
            LSTM(64, input_shape=(SEQ_LEN, len(FEATURES)), return_sequences=True),
            BatchNormalization(),
            Dropout(0.15),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.15),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(
            loss=tf.keras.losses.Huber(delta=1.0),
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4)
        )
        hist = model.fit(
            X_3d, y_arr,
            epochs=500,
            batch_size=16,
            validation_split=0.2,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=40,
                    restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', patience=15, factor=0.5,
                    min_lr=1e-6, verbose=0),
            ],
            verbose=0
        )

        cluster_models[cid]  = model
        cluster_scalers[cid] = scaler

        epoch_berhenti   = len(hist.history['loss'])
        val_loss_terbaik = min(hist.history['val_loss'])
        print(
            f"[WORKER]   LSTM klaster {cid} selesai: {len(df_cl)} sampel, "
            f"berhenti di epoch {epoch_berhenti}/500, "
            f"val_loss terbaik={val_loss_terbaik:.4f}",
            flush=True
        )

    for cid, model in cluster_models.items():
        model_path = output_prefix + f'_cluster_{cid}.h5'
        model.save(model_path)
        print(f"[WORKER]   Model klaster {cid} disimpan: {model_path}", flush=True)

    meta = {
        'cids':    list(cluster_models.keys()),
        'scalers': cluster_scalers,
    }
    joblib.dump(meta, output_prefix + '_meta.pkl')
    print(f"[WORKER]   Meta disimpan: {output_prefix}_meta.pkl", flush=True)

    print("[LSTM_WORKER_DONE]", flush=True)

if __name__ == '__main__':
    main()
