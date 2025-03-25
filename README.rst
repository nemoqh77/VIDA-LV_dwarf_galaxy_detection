PhotCalib
-----------

Calibration tool for Prisitine CaHK data. For more details see Martin, Starkenburg & Yuan 2023



Installation
----------------

Before installation, make sure you have pytorch or install essential dependencies :

.. code::

  pip install -r requirements.txt



* Installation

.. code::

  pip install --user -e.

* Manual installation

download the repository and run the setup

.. code::

  python setup.py install      

Getting started 
----------------

.. code::

  cd examples
 

To Calibrate CaHK data using the run model trained by the main survey:
.. code::

  python calib_raw.py 17Am05 combined_catalogue_17Am05_sel.fits -D cpu
 
Substitute 17Am05 to the run you need and make sure the input data for calibration has the same format as that in the examples/data/.fits

* D (str) is the device (default:cpu, gpu) to run pytorch. If gpu is chosen, it will use the first graphic card cuda:0 by default.


To train calibration run model:
.. code::

  python calib_mod.py 17Am05

Substitute 17Am05 to the run for calibration with the input file as examples/data/inputs_run.npy

Specify the device and training parameters (the followings are the default settings)
.. code::

  python calib_mod.py 17Am05 -D cpu -lr 1e-6 -n 400 -mom 0.9 -thr 1e-2

* D (str) is the device (default:cpu, mps, gpu) to run pytorch. 
* n (int) is the number of training epochs.
* lr (float) is the initial learning rate, which will decrease using scheduler (see details at https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau).
* thr (float) is the threshold of the scheduler.
* mom (float) is the momentum of gradient descent.

Citing this work
----------------

.. code::

  coming soon
