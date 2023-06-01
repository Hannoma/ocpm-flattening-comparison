import os

DATASETS = {
    'Order_Process': {
        'filename': 'order_process',
        'format': 'jsonocel',
        'columns': {
            'start_timestamp': 'event_start_timestamp',
            'resource': None,
            'object_columns': ['order', 'item', 'delivery'],
            'value_columns': {},
        },
    },
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

WORKER_COUNT = 2
