"""Els números del gradient boosting.

Punt de partida i no els valors per defecte de la llibreria, que arrenca en 100
arbres i taxa 0,1.
"""
WINDOW = 1085
TREES = 500
RATE = 0.05
LEAVES = 31

# Arbres sense millorar en alguna de les parts de la validació abans d'aturar
# l'entrenament de la part B. Trebbien et al. (2023) en diuen "a specific number of
# epochs" sense el número: és inventat, sense cercar, fins que la cerca el triï.
PATIENCE = 50
