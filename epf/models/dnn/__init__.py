"""La xarxa neuronal, la xarxa de punt de Marcjasz et al. (2023)."""

import os

# Sense les línies d'informació que TensorFlow escriu en carregar-se, abans que cap
# mòdul de la xarxa importi Keras; els avisos i els errors hi queden.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
