# Manuel de maintenance — Pompe centrifuge P-200 (Rév. C)

## 1 Généralités

Ce manuel s'applique aux pompes centrifuges P-200 installées sur les circuits de refroidissement.
Il remplace la révision B. Les valeurs de ce manuel priment sur toute fiche antérieure.

## 2 Sécurité

### 2.1 Consignation

Avant toute intervention, consigner électriquement la pompe (sectionneur Q12 ouvert et cadenassé)
puis hydrauliquement (vannes V1 et V2 fermées, circuit purgé). Vérifier l'absence de pression au
manomètre PI-201 : il doit indiquer 0 bar.

Danger : ne jamais desserrer les brides tant que le circuit est sous pression.

### 2.2 Équipements de protection

Gants anti-coupure, lunettes et chaussures de sécurité. Casque antibruit obligatoire si une pompe
voisine est en service (niveau sonore supérieur à 85 dB(A)).

## 3 Maintenance préventive

### 3.1 Périodicités

| Opération | Périodicité | Durée |
|---|---|---|
| Contrôle visuel des fuites | Hebdomadaire | 10 min |
| Graissage des roulements | Toutes les 2 000 h | 20 min |
| Mesure vibratoire | Mensuelle | 15 min |
| Remplacement de la garniture mécanique | 16 000 h ou fuite > 10 gouttes/min | 3 h |

### 3.2 Graissage des roulements

Utiliser une graisse lithium NLGI 2 (référence GR-L2). Quantité : 12 g côté accouplement,
8 g côté opposé. Ne pas sur-graisser : un excès de graisse provoque un échauffement du palier.

## 4 Couples de serrage

Serrer à froid, en croix, en deux passes (50 % puis 100 % du couple).

| Élément | Vis | Couple (N·m) |
|---|---|---|
| Brides d'aspiration | M16 | 120 |
| Brides de refoulement | M12 | 60 |
| Carter de pompe | M10 | 49 |
| Pattes moteur | M12 | 70 |

## 5 Dépannage

### 5.1 Codes alarme

| Code | Signification | Action |
|---|---|---|
| E-12 | Température palier > 85 °C | Vérifier le graissage (§ 3.2) |
| E-41 | Vibration > 7,1 mm/s | Contrôler l'alignement, puis les roulements |
| E-57 | Pression d'aspiration basse | Vérifier la vanne V1 et le filtre F-3 |

### 5.2 Vibrations excessives (alarme E-41)

1. Consigner la pompe (§ 2.1).
2. Contrôler l'alignement moteur-pompe au laser : tolérance 0,05 mm.
3. Si l'alignement est correct, mesurer le spectre vibratoire : une raie à la fréquence de défaut
   de bague extérieure indique un roulement endommagé.
4. Remplacer les roulements 6309-2Z (côté accouplement) et 6307-2Z (côté opposé).
