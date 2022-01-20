Usage
=====

.. _installation:

Installation
------------

To use PyMSM, first install it using pip:

.. code-block:: console

   (.venv) $ pip install PyMSM

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``pymsm.multi_state_competing_risks_model.MultiStateModel()`` class:


The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

