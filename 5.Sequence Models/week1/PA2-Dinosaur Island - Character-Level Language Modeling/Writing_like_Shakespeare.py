from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

generate_output()




""" Results :

after 1000 + 100 epochs 
Here is your poem: 

No one can live without love,
yet of which will ath, thou ulms ne thy trumy,
which sigh to my dour, on tae, and there
thing, ard save herpernawings sweet self?
o year doth gies yee in she beaind, regiping,
but berow'st still me a will in alpost lever?
who guck me honol usutn comaly thought
on thin my gowe dearth agut thy uepars jught,
but shall mistles is lha besall ares in see.



the fawst are makgen this lyoud to gike thi
"""