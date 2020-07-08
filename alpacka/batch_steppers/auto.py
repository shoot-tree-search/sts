"""Environment steppers."""

import os

from alpacka import batch_steppers

if 'LOCAL_RUN' in os.environ:
    class AutoBatchStepper(batch_steppers.LocalBatchStepper):
        pass
else:
    class AutoBatchStepper(batch_steppers.RayBatchStepper):
        pass
