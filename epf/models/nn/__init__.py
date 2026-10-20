"""Les xarxes de Marcjasz et al. (2023), les dues distribucionals i la de punt."""

import os

# Sense les línies d'informació que TensorFlow escriu en carregar-se, abans que cap
# mòdul de les xarxes importi Keras; els avisos i els errors hi queden.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
