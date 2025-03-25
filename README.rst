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

Getting started 
----------------

.. code::

  cd examples

Download the ViT model "vit-base-patch16-224-in21k"
----------------

https://huggingface.co/docs/transformers/model_doc/vit



Train model with dataset sample A and B:
----------------
.. code::

  python3 ViT_train.py 1 1 0.00005 
 
The first "1" means run train sampleA , second "1" means run train sampleB (if don't train the sampleA or B, change "1" to "0")

0.00005: learning rate


To train the model with dataset sample A and B:
.. code::

  python3 ViT_test.py 1 1 0.00005 ep1 ep2
 
The first "1" means run train sampleA , second "1" means run train sampleB (if don't train the sampleA or B, change "1" to "0")

0.00005: learning rate

ep1 and ep2 represent the epoch range (0-60) you selected for model testing to obtain prediction results. 



Citing this work
----------------

.. code::

  coming soon
