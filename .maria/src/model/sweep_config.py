sweep_config = {
    "method": "grid",
    "name": "audio_cnn_sweep",
    "metric": {
        "name": "val_acc",  # Changed to validation accuracy
        "goal": "maximize"
    },
    "parameters": {
        "LEARNING_RATE": {
            "values": [0.001, 0.01, 0.0001]
        },
        "BATCH_SIZE": {
            "values": [2, 4, 8]
        },
        "DROPOUT_RATE": {
            "values": [0.1, 0.3, 0.5]
        }
    }
}
