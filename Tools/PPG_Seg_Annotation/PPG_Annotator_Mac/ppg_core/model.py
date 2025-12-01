import numpy as np
import os

class SignalModel:
    def __init__(self):
        self.data = None
        self.fs = 100 
        self.filepath = None
        self.filename = None
        self.is_segmented = False
        self.original_shape = None
        self.current_win_size = 256
        self.current_stride = 128

    def load_from_file(self, filepath, win_size=None, stride=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        
        try:
            raw_data = np.load(filepath, mmap_mode='r')
        except Exception as e:
            raise e

        self.original_shape = raw_data.shape
        
        # Segmented Data Logic
        if raw_data.ndim >= 2:
            self.is_segmented = True
            
            # Determine params (User input > Inferred > Default)
            _win = win_size if win_size else raw_data.shape[-1]
            _stride = stride if stride else (_win // 2)
            
            self.current_win_size = _win
            self.current_stride = _stride

            # Reconstruct
            segments = raw_data.reshape(-1, _win)
            if segments.shape[0] > 0:
                body = segments[:-1, :_stride].flatten()
                tail = segments[-1, :].flatten()
                self.data = np.concatenate([body, tail])
            else:
                self.data = segments.flatten()
            self.data = self.data.astype(np.float32)
        else:
            self.is_segmented = False
            self.data = raw_data.astype(np.float32).reshape(-1)

    def get_data(self):
        return self.data
