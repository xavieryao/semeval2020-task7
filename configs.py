original_lstm = {
    'hidden': 16,
    'dropout': 0.5,
    'linear': 64,
    'norm': False,
    'task': 'regression'
}

original_lstm_with_norm = {
    'hidden': 16,
    'dropout': 0.5,
    'linear': 64,
    'norm': True,
    'task': 'regression'
}

larger_lstm = {
    'hidden': 64,
    'dropout': 0.5,
    'linear': 64,
    'norm': True,
    'task': 'regression'
}

larger_lstm_5bin = {
    'hidden': 128,
    'linear': 128,
    'dropout': 0.5,
    'norm': False,
    'task': '5-classification'
}

original_lstm_5bin = {
    'hidden': 16,
    'dropout': 0.5,
    'linear': 64,
    'norm': False,
    'task': '5-classification'
}

original_lstm_3bin = {
    'hidden': 16,
    'dropout': 0.5,
    'linear': 64,
    'norm': False,
    'task': '3-classification'
}