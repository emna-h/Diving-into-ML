# Synthèse d'un réseau d'antennes circulaire par réseau antagoniste génératif conditionnel (cGAN)

## Partie théorique : mathématiques, principe de fonctionnement et étapes du modèle

> **Portée de ce document.** Il décrit *comment le modèle fonctionnera* : les mathématiques de chaque composant et chaque étape qui mène d'une spécification à une architecture d'antennes. Il ne contient ni code, ni résultats d'entraînement. Il sert de base à la construction du modèle.

---

## 0. Vue d'ensemble

**Objectif.** À partir de spécifications de rayonnement souhaitées, trouver le nombre d'antennes à placer sur chacun des 5 anneaux d'un réseau circulaire.

```
 Spécifications souhaitées            Configuration cherchée
 c = [gain, SSL, HPBW, θ₀]   ───▶    x = [N₁, N₂, N₃, N₄, N₅]
       (4 nombres)                         (5 entiers)
```

**Idée centrale.** Le sens x → c est une **fonction connue** (la physique, section 1). Le sens c → x est un **problème inverse** sans formule. On l'aborde avec un modèle génératif qui apprend la distribution `p(x | c)` à partir d'exemples (c, x) fabriqués avec la physique.

**Les deux phases du modèle.**

```
 PHASE D'APPRENTISSAGE (une fois)
   Physique ──▶ jeu de données (c, x) ──▶ le cGAN apprend p(x | c)

 PHASE D'UTILISATION (à chaque demande)
   c souhaité ─┐
               ├─▶ Générateur ─▶ K candidats ─▶ arrondi ─▶ Physique ─▶ choix du meilleur
   bruit z ────┘                                          (vérification)
```

Le réseau de neurones **propose** ; la physique **vérifie et départage**. Les sections suivantes détaillent chaque bloc.

---

## 1. La physique : du réseau d'antennes au diagramme de rayonnement

### 1.1 Paramètres fixes

| Grandeur | Valeur |
|---|---|
| Fréquence porteuse f | 2,45 GHz |
| Célérité c | 3·10⁸ m/s |
| Longueur d'onde λ = c / f | 0,12245 m |
| Nombre d'onde k = 2π / λ | 51,31 rad/m |
| Nombre d'anneaux | 5 |
| Rayon de l'anneau i (i = 0…4) | `aᵢ = r₀ + i·Δr`, avec r₀ = 0,2 λ et Δr = 0,5 λ |
| Éléments par anneau | `Nᵢ ∈ {0, 1, …, 10}`, au moins un anneau non vide |

Les rayons valent donc 0,2 ; 0,7 ; 1,2 ; 1,7 ; 2,2 λ, soit **0,0245 ; 0,0857 ; 0,1469 ; 0,2082 ; 0,2694 m**.

### 1.2 Géométrie

Les Nᵢ éléments de l'anneau *i* sont régulièrement espacés. Pour n = 0, …, Nᵢ − 1 :

```
φₙ = 2π n / Nᵢ          xₙ = aᵢ cos φₙ          yₙ = aᵢ sin φₙ
```

Chaque élément est repéré par son rayon aᵢ et son angle φₙ. L'ensemble de la configuration est le vecteur **x = [N₁, …, N₅]**.

### 1.3 Facteur de réseau

Les éléments sont supposés **isotropes** et excités avec une amplitude unitaire. Le diagramme du réseau est alors égal à son **facteur de réseau** (AF) : la somme des ondes émises par tous les éléments, avec leurs déphasages géométriques.

Pour un élément de rayon aₙ et d'angle φₙ, observé dans la direction (θ, φ), avec une phase d'excitation αₙ :

```
AF(θ, φ) = Σₙ exp( j [ k aₙ sinθ cos(φ − φₙ) + αₙ ] )
```

Le terme `k aₙ sinθ cos(φ − φₙ)` est le déphasage dû à la position de l'élément : l'onde de l'élément n parcourt un chemin plus court ou plus long que celle de l'origine selon la direction d'observation.

### 1.4 Pointage du faisceau

Pour orienter le lobe principal vers (θ₀, φ₀), on compense le déphasage géométrique dans cette direction :

```
αₙ = − k aₙ sinθ₀ cos(φ₀ − φₙ)
```

En remplaçant αₙ, on obtient la forme utilisée dans tout le projet :

```
AF(θ) = Σₙ exp( j k aₙ [ sinθ cos(φ − φₙ) − sinθ₀ cos(φ₀ − φₙ) ] )
```

**Pourquoi le pic est en θ₀.** Pour θ = θ₀ (et φ = φ₀), le crochet s'annule : tous les termes valent exp(0) = 1 et s'additionnent **en phase**. On obtient :

```
|AF(θ₀)| = N_total = Σᵢ Nᵢ
```

C'est la valeur maximale possible. Dans toute autre direction, les termes sont déphasés et s'annulent partiellement.

**Coupe utilisée.** On fixe φ = φ₀ = 0 et on balaie θ de 0 à 2π sur 1000 points (pas de 0,36°). C'est une coupe du diagramme dans le plan contenant les axes x et z. Comme sinθ = sin(π − θ), le diagramme est **symétrique** : il possède **deux lobes principaux**, en θ₀ et en 180° − θ₀.

### 1.5 Les trois grandeurs de performance

On normalise d'abord le diagramme par son maximum :

```
AF_norm(θ) = |AF(θ)| / max|AF|          AF_dB(θ) = 20 log₁₀ AF_norm(θ)     (borné à −40 dB)
```

| Grandeur | Définition | Remarque |
|---|---|---|
| **Gain du lobe principal** | `G = 20 log₁₀( max|AF| )` en dB | Vaut `20 log₁₀(N_total)` : il ne dépend que du **nombre total d'éléments**. |
| **HPBW** | Largeur angulaire du lobe principal entre les points à −3 dB | Plus il est petit, plus le faisceau est étroit. Vaut 180° par convention si un côté n'atteint jamais −3 dB. |
| **SSL** | Niveau (dB, relatif au lobe principal) du plus haut pic situé à plus de 1 dB sous le lobe principal | Négatif ; plus il est négatif, meilleure est la suppression des lobes secondaires. |

Le vecteur de performances est donc **c = [G, SSL, HPBW, θ₀]**. La physique définit ainsi une fonction

```
Φ : (x, θ₀)  ⟶  (G, SSL, HPBW)
```

### 1.6 Exemple numérique de référence (fil rouge)

Prenons **x = [2, 9, 4, 5, 5]** et **θ₀ = 45°**.

- Nombre total d'éléments : 2 + 9 + 4 + 5 + 5 = **25**.
- Anneau 2 (N = 9) : angles φₙ = 0°, 40°, 80°, 120°, …, 320°. Phase de pointage du premier élément : α₀ = −k a₁ sin 45° cos 0° = −51,31 × 0,0857 × 0,7071 = −3,11 rad ≈ **−178,2°**.
- Résultats de la physique : **gain = 27,96 dB** (= 20 log₁₀ 25), **HPBW = 22,34°**, **SSL = −8,02 dB**, lobe principal à **45,05°** (et son symétrique à 135°).

Cet exemple sera réutilisé tout au long du document : c'est un couple (x, c) réel, avec c = [27,96 ; −8,02 ; 22,34 ; 45].

---

## 2. Formulation du problème inverse

### 2.1 Sens direct et sens inverse

Le sens direct est une **fonction** : à (x, θ₀) correspond un seul triplet de performances. Le sens inverse ne l'est pas forcément : plusieurs configurations x peuvent produire des performances égales ou très voisines. Le problème inverse est alors **un-à-plusieurs** (« mal posé » au sens d'Hadamard : la solution n'est pas unique).

### 2.2 Régression déterministe (rappel du FFNN)

Un réseau feed-forward apprend une fonction `x̂ = f_θ(c)`, en minimisant l'erreur quadratique :

```
L_MSE = (1/M) Σᵢ ‖ f_θ(cᵢ) − xᵢ ‖²
```

Le minimiseur théorique de cette perte est la **moyenne conditionnelle** `f*(c) = E[x | c]`. Si deux solutions distinctes x_A et x_B existent pour une même spécification, le régresseur tend vers (x_A + x_B)/2, qui n'est pas nécessairement une solution. Un régresseur fournit en outre **une seule** réponse. Cette approche n'est pas incorrecte (elle fonctionne bien quand la solution est quasi unique) ; sa limite est de ne pas décrire l'ensemble des solutions.

### 2.3 Formulation générative

On modélise plutôt la **loi conditionnelle** :

```
x  ~  p(x | c)
```

Au lieu d'une réponse, on dispose d'un *échantillonneur* qui produit, pour une spécification donnée, des configurations plausibles et variées. La physique Φ sert ensuite à retenir la meilleure.

---

## 3. Les données

### 3.1 Construction

Chaque exemple est fabriqué avec la physique (aucune mesure réelle) :

1. Tirer une architecture `x` uniformément dans {0,…,10}⁵, en rejetant le tirage tout nul.
2. Tirer `θ₀ ~ U(0°, 180°)`.
3. Calculer `(G, SSL, HPBW) = Φ(x, θ₀)` avec la section 1.
4. Enregistrer la paire : condition `c = [G, SSL, HPBW, θ₀]` et configuration réelle `x`.

On obtient N = 50 000 paires `(cᵢ, xᵢ)` réparties en deux matrices : `Minput` (4 × N, les conditions) et `Moutput` (5 × N, les configurations). **Le cGAN apprend `p(x | c)` à partir de ces paires** : x est ce que l'on veut générer, c est ce qui conditionne la génération.

### 3.2 Normalisation

Les réseaux de neurones apprennent mieux avec des entrées et des sorties de même ordre de grandeur. On normalise donc :

```
condition :    c_norm,j = ( c_j − min_j ) / ( max_j − min_j )        j = 1…4
configuration: x_norm,i = Nᵢ / 10                                      i = 1…5
```

Les min et max sont calculés sur le **jeu d'entraînement uniquement** puis **réutilisés à l'identique** à l'utilisation (sinon le modèle verrait des échelles différentes de celles de l'apprentissage).

Pour l'exemple de la section 1.6 (min/max du jeu complet) : `c_norm ≈ [0,823 ; 0,793 ; 0,047 ; 0,250]` et `x_norm = [0,2 ; 0,9 ; 0,4 ; 0,5 ; 0,5]`.

### 3.3 Découpage sans fuite

Les données sont réparties en **apprentissage (80 %)**, **validation (10 %)** et **test (10 %)**.

*Piège à éviter.* L'architecture est tirée parmi 11⁵ = 161 051 possibilités. Avec 50 000 tirages, environ **7 000 exemples répètent une architecture déjà vue** (avec un autre θ₀). Un découpage aléatoire simple mettrait une même architecture dans l'apprentissage et dans le test, et le test mesurerait en partie de la mémorisation. On répartit donc des **groupes entiers** : tous les exemples ayant le même vecteur [N₁…N₅] vont dans le même sous-ensemble. Le découpage est reproductible (graine fixe).

| Sous-ensemble | Rôle |
|---|---|
| Apprentissage | Ajuster les poids des réseaux |
| Validation | Choisir le meilleur état du modèle pendant l'apprentissage |
| Test | Évaluation finale, jamais utilisée pour décider quoi que ce soit |

---

## 4. Les réseaux antagonistes génératifs (GAN)

### 4.1 Les deux joueurs

- **Générateur G** : fonction paramétrée qui transforme un bruit `z ~ N(0, I)` en un échantillon `x̃ = G(z)`.
- **Discriminateur D** : fonction paramétrée qui renvoie `D(x) ∈ [0, 1]`, la probabilité que x soit **réel** (issu du jeu de données) plutôt que généré.

**Rôle du bruit z.** Une fonction déterministe de c produirait toujours la même sortie. Le bruit z est la **source d'aléa** : en tirant plusieurs z, G produit plusieurs sorties différentes. G transforme ainsi la loi simple `N(0, I)` en la loi complexe des données : on dit qu'il définit une loi `p_G`.

### 4.2 Le jeu minimax

```
min_G max_D  V(D, G) = E_{x ~ p_data}[ log D(x) ]  +  E_{z ~ N(0,I)}[ log( 1 − D(G(z)) ) ]
```

- D **maximise** V : il veut donner une probabilité proche de 1 aux vrais échantillons (premier terme) et proche de 0 aux faux (second terme).
- G **minimise** V : il veut que D donne une probabilité élevée à ses échantillons.

### 4.3 Discriminateur optimal et signification du jeu

Pour un G fixé, le meilleur discriminateur possible est :

```
D*(x) = p_data(x) / ( p_data(x) + p_G(x) )
```

En le réinjectant dans V, on obtient :

```
V(D*, G) = − log 4 + 2 · JS( p_data ‖ p_G )
```

où JS est la divergence de Jensen–Shannon (toujours ≥ 0, et nulle si et seulement si les deux lois sont égales). **Minimiser V par rapport à G revient donc à rapprocher p_G de p_data.** À l'optimum théorique, p_G = p_data et D*(x) = 1/2 partout : D ne sait plus distinguer le vrai du faux.

### 4.4 Pertes pratiques (entropie croisée binaire)

Avec l'entropie croisée binaire `BCE(p, t) = −[ t log p + (1 − t) log(1 − p) ]` :

```
Perte du discriminateur :   L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)

Perte du générateur (non saturante) :   L_G = BCE(D(G(z)), 1) = − log D(G(z))
```

**Pourquoi la forme non saturante.** La forme d'origine de G, `log(1 − D(G(z)))`, a un gradient très faible au début de l'apprentissage, quand D rejette facilement les faux (D(G(z)) ≈ 0). La forme `−log D(G(z))` a le même point d'équilibre mais un gradient exploitable dès le départ.

---

## 5. Le GAN conditionnel (cGAN)

### 5.1 Principe

On fournit la **condition c** au générateur *et* au discriminateur :

```
x̃ = G(z, c)                      D(x, c) ∈ [0, 1]

min_G max_D  V = E_{(x,c) ~ p_data}[ log D(x, c) ] + E_{z, c}[ log( 1 − D(G(z, c), c) ) ]
```

Le raisonnement de la section 4.3 s'applique à chaque c fixé : à l'équilibre, `p_G(x | c) = p_data(x | c)`. Le générateur devient un échantillonneur de la loi conditionnelle de la section 2.3.

### 5.2 Comment la condition entre dans les réseaux

Par **concaténation** avec l'entrée :

```
Entrée de G :  u = [ z ; c ]  ∈ ℝ^(16+4) = ℝ²⁰
Entrée de D :  v = [ x ; c ]  ∈ ℝ^(5+4)  = ℝ⁹
```

- Pour G : « invente une configuration pour *cette* spécification ».
- Pour D : « cette configuration est-elle une réponse crédible pour *cette* spécification ? » D juge donc la **cohérence de la paire (x, c)**, pas seulement le réalisme de x.

### 5.3 Discriminateur sensible à la cohérence (« matching-aware »)

Si D ne voit que (x réel, c) et (x généré, c), rien ne l'oblige à vérifier que x correspond bien à c : il peut se contenter de juger si x ressemble à une architecture quelconque. On ajoute donc un troisième type d'exemple négatif : un x **réel** associé à une condition **d'un autre échantillon**, c̃ (obtenue en mélangeant les conditions d'un lot) :

```
L_D = BCE( D(x, c), 0,9 ) + ½ [ BCE( D(G(z,c), c), 0 ) + BCE( D(x, c̃), 0 ) ]
```

Ainsi D apprend deux choses : repérer les faux, et repérer les paires incohérentes. C'est ce second apprentissage qui force G à **tenir compte de c**.

### 5.4 Deux stabilisations

**Label smoothing.** La cible « réel » vaut 0,9 au lieu de 1. D n'est ainsi jamais poussé vers une certitude absolue, ce qui limite sa tendance à écraser G.

**Dé-quantification.** Les vraies configurations sont des **entiers**, alors que G produit des **réels** dans [0, 1]. D pourrait donc distinguer le vrai du faux uniquement parce que les valeurs réelles tombent exactement sur la grille {0, 0,1, …, 1}. Pour l'empêcher, pendant l'apprentissage on remplace chaque vraie valeur par :

```
x_cont = clip( (N + u) / 10 , 0 , 1 ),     u ~ U(−0,5 ; +0,5)
```

La loi des données devient continue, comparable à celle de G. L'information est conservée : `round(10 · x_cont) = N` exactement.

---

## 6. Architecture des réseaux

### 6.1 Dimensions

| Objet | Dimension |
|---|---|
| Condition c (normalisée) | 4 |
| Bruit z | 16 |
| Entrée de G = [z ; c] | 20 |
| Sortie de G (x̃ normalisé) | 5 |
| Entrée de D = [x ; c] | 9 |
| Sortie de D (logit) | 1 |

### 6.2 Calcul du générateur

Réseau à couches denses :

```
h₁ = φ( W₁ u + b₁ )      ∈ ℝ¹²⁸
h₂ = φ( W₂ h₁ + b₂ )     ∈ ℝ²⁵⁶
h₃ = φ( W₃ h₂ + b₃ )     ∈ ℝ¹²⁸
x̃  = σ( W₄ h₃ + b₄ )     ∈ [0, 1]⁵
```

avec :
- `φ(t) = max(0,2 t ; t)` : LeakyReLU. Contrairement à la ReLU, elle garde une pente non nulle pour t < 0, ce qui évite que des neurones ne s'éteignent définitivement (problème fréquent dans les GAN).
- `σ(t) = 1 / (1 + e^(−t))` : sigmoïde. Elle ramène la sortie dans [0, 1] et impose ainsi, par construction, la contrainte `0 ≤ Nᵢ ≤ 10`.

Nombre de paramètres : 69 253.

### 6.3 Calcul du discriminateur

```
h₁ = Dropout( φ( W₁ v + b₁ ) )     ∈ ℝ¹²⁸
h₂ = Dropout( φ( W₂ h₁ + b₂ ) )    ∈ ℝ¹²⁸
h₃ = Dropout( φ( W₃ h₂ + b₃ ) )    ∈ ℝ⁶⁴
ℓ  = W₄ h₃ + b₄                    ∈ ℝ         (logit)
D(x, c) = σ(ℓ)
```

Le *dropout* (probabilité 0,3) met aléatoirement à zéro 30 % des neurones à chaque passage et affaiblit D, qui apprend vite sur un problème de si petite dimension. Nombre de paramètres : 26 113.

Dans la pratique, la perte BCE est calculée directement sur le logit ℓ (forme numériquement stable, équivalente à appliquer σ puis BCE).

---

## 7. Mécanisme d'apprentissage

### 7.1 Une itération sur un mini-lot

On prélève B = 128 paires (c, x) du jeu d'apprentissage. Chaque itération comporte deux mises à jour, **alternées**.

**Étape 1 : préparer les données.**
- `x_réel` = dé-quantification de x (section 5.4).
- `z ~ N(0, I)`, de dimension 16, un par exemple du lot.
- `x̃ = G(z, c)`.
- `c̃` = conditions mélangées du lot.

**Étape 2 : mettre à jour D.** Calculer `L_D` (section 5.3) avec `x̃` « figé » (on ne propage pas le gradient dans G) puis ajuster uniquement les paramètres de D.

**Étape 3 : mettre à jour G.** Calculer `L_G = BCE(D(x̃, c), 1)` avec le D qui vient d'être mis à jour, puis ajuster uniquement les paramètres de G.

### 7.2 Comment G apprend à travers D

G n'est jamais comparé directement à une « bonne réponse ». Son gradient passe **à travers D** (règle de la chaîne) :

```
∂L_G/∂θ_G  =  ∂L_G/∂D  ·  ∂D/∂x̃  ·  ∂x̃/∂θ_G
```

Le terme `∂D/∂x̃` indique, pour chaque sortie x̃, dans quelle direction la déplacer pour que D la juge plus crédible. G ajuste alors ses poids dans cette direction. **D joue le rôle d'une fonction de perte apprise.**

*Conséquence importante.* Ce chaînage exige que tout soit dérivable de G à la perte. Le modèle physique ne l'est pas (arrondi, détection de pics, seuils) : il n'est donc **pas** placé dans la boucle d'apprentissage. Il intervient uniquement pour vérifier, sélectionner et évaluer.

### 7.3 Mise à jour des poids (Adam)

Pour un paramètre w de gradient g_t à l'itération t :

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t              (moyenne mobile du gradient)
v_t = β₂ v_{t−1} + (1 − β₂) g_t²             (moyenne mobile du carré du gradient)
m̂_t = m_t / (1 − β₁ᵗ)      v̂_t = v_t / (1 − β₂ᵗ)      (correction du biais initial)
w ← w − η · m̂_t / ( √v̂_t + ε )
```

Réglages usuels des GAN : pas d'apprentissage η = 2·10⁻⁴, β₁ = 0,5 (inertie réduite, pour réagir vite à l'adversaire qui change), β₂ = 0,999.

### 7.4 Algorithme complet

```
Entrées : jeu (cᵢ, xᵢ) normalisé ; hyperparamètres
Pour chaque epoch :
    Pour chaque mini-lot (c, x) de taille B :
        1. x_réel ← dé-quantification(x) ;  z ~ N(0, I) ;  x̃ ← G(z, c) ;  c̃ ← mélange(c)
        2. L_D ← BCE(D(x_réel, c), 0,9) + ½[ BCE(D(x̃, c), 0) + BCE(D(x_réel, c̃), 0) ]
           mise à jour de D
        3. L_G ← BCE(D(x̃, c), 1)
           mise à jour de G
    Toutes les quelques epochs :
        pour des conditions de VALIDATION : générer, arrondir, évaluer avec la physique
        conserver le modèle dont l'erreur physique de validation est la plus faible
```

**Critère de choix du modèle.** Les pertes d'un GAN oscillent par construction (G et D progressent l'un contre l'autre) : elles ne mesurent pas la qualité. Le modèle retenu est celui qui minimise l'**erreur physique de validation** (section 9).

**Signe d'un entraînement équilibré.** `D(x réel, c)` et `D(x̃, c)` restent voisins de 0,5 : aucun des deux joueurs n'écrase l'autre.

---

## 8. Utilisation du modèle : de la demande au résultat, pas à pas

Une fois le modèle appris, voici **toutes les étapes** qui produisent une architecture. On les suit avec l'exemple c = [27,96 ; −8,02 ; 22,34 ; 45].

**Étape 1 : saisir la spécification.**
`c = [gain, SSL, HPBW, θ₀]`. On vérifie que chaque valeur est dans la plage vue à l'apprentissage ; en dehors, le modèle extrapole et le résultat est peu fiable.

**Étape 2 : normaliser.**
`c_norm = (c − min) / (max − min)` avec les paramètres sauvegardés à l'apprentissage. Exemple : `[0,823 ; 0,793 ; 0,047 ; 0,250]`.

**Étape 3 : tirer K bruits.**
`z₁, …, z_K ~ N(0, I)`, chacun dans ℝ¹⁶, par exemple K = 50. Chaque z_k donnera un candidat différent.

**Étape 4 : générer.**
Pour chaque k : `u_k = [z_k ; c_norm]` (20 valeurs) → `x̃_k = G(u_k) ∈ [0, 1]⁵`.

**Étape 5 : dénormaliser et arrondir.**
`x_k = clip( round(10 · x̃_k), 0, 10 )`. Exemple : une sortie `[0,18 ; 0,93 ; 0,41 ; 0,52 ; 0,49]` donne `[2, 9, 4, 5, 5]`.

**Étape 6 : appliquer la contrainte « au moins un anneau actif ».**
Si l'arrondi donne `[0, 0, 0, 0, 0]`, on active un élément sur l'anneau dont la sortie continue était la plus grande (contrainte héritée de la construction des données).

**Étape 7 : calculer la physique de chaque candidat.**
Pour chaque x_k : position des éléments (1.2) → facteur de réseau avec pointage en θ₀ (1.4) → diagramme → `(G_k, SSL_k, HPBW_k)` (1.5).

**Étape 8 : mesurer l'écart avec la demande.**
Voir la section 9 : une erreur globale `E_k` par candidat.

**Étape 9 : choisir.**
`k* = argmin_k E_k`. On retient `x_{k*}`. On peut aussi présenter les meilleurs candidats **distincts** : si plusieurs architectures satisfont la demande, le concepteur a le choix (par exemple selon le nombre total d'éléments).

**Étape 10 : restituer.**
Architecture `[N₁ … N₅]`, performances réellement obtenues (calculées par la physique, pas prédites par le réseau) et diagramme de rayonnement.

**Schéma récapitulatif :**

| Étape | Entrée | Sortie | Équation |
|---|---|---|---|
| 2 | c (4) | c_norm (4) | `(c − min)/(max − min)` |
| 3 | : | z_k (16), k = 1…K | `N(0, I)` |
| 4 | [z_k ; c_norm] (20) | x̃_k (5) | `G(z, c)` |
| 5-6 | x̃_k | x_k entiers (5) | `round(10 x̃)`, clip, réparation |
| 7 | x_k, θ₀ | G_k, SSL_k, HPBW_k | `AF(θ)` (1.4) puis 1.5 |
| 8 | (G_k, SSL_k, HPBW_k), c | E_k | formule de la section 9 |
| 9 | E_1…E_K | x_{k*} | `argmin` |

---

## 9. Évaluation de la qualité

### 9.1 Pourquoi pas seulement la perte du GAN

La perte d'un GAN n'a pas de sens absolu (elle dépend de l'adversaire à l'instant t). Le critère pertinent est la **qualité physique** : le réseau d'antennes proposé a-t-il bien les performances demandées ?

### 9.2 Erreurs par composante

Pour une configuration générée x̂, on calcule `(Ĝ, SSL̂, HPBŴ) = Φ(x̂, θ₀)` puis les écarts avec la demande :

```
e_gain = | Ĝ − G_demandé |        e_SSL = | SSL̂ − SSL_demandé |        e_HPBW = | HPBŴ − HPBW_demandé |
```

L'erreur sur la direction du lobe principal tient compte de la symétrie : c'est la plus petite erreur circulaire par rapport à θ₀ **et** à 180° − θ₀.

### 9.3 Erreur globale pondérée

```
E = 0,33 · 100 · e_gain / |G_max|  +  0,33 · 100 · e_SSL / |SSL_max|  +  0,34 · 100 · e_HPBW / 180        (en %)
```

Chaque écart est rapporté à une échelle de référence (maxima du jeu d'apprentissage pour le gain et le SSL, 180° pour le HPBW) pour pouvoir additionner des grandeurs d'unités différentes. Les poids 0,33 / 0,33 / 0,34 donnent la même importance aux trois spécifications.

### 9.4 Modes et contrôles d'évaluation

| Mode | Principe | Ce qu'il apprend |
|---|---|---|
| **Un tirage (K = 1)** | Un seul z par demande | Qualité d'une réponse typique |
| **Meilleur de K** | K tirages, on garde le plus proche de la demande | Intérêt de l'approche générative |
| **Condition mélangée** | G reçoit la condition d'un autre échantillon, l'erreur est mesurée par rapport à la vraie demande | Si l'erreur est nettement plus grande, G **utilise vraiment c** |
| **Architecture aléatoire** | Tirage uniforme sans modèle | Niveau de référence « sans intelligence » |

On mesure aussi la **diversité** : le nombre d'architectures distinctes obtenues sur K tirages, et la fréquence à laquelle l'architecture de référence du jeu de test est retrouvée. Une architecture différente de la référence mais aux performances correctes n'est **pas** une erreur : elle illustre la nature un-à-plusieurs du problème.

Enfin, on compare visuellement le **diagramme de rayonnement de la référence** et celui de la configuration générée (polaire et cartésien).

---

## 10. Hypothèses et limites de la méthode

1. **Physique simplifiée** : éléments isotropes, excitations d'amplitude unitaire, pas de couplage mutuel, une seule coupe 2D du diagramme. Les conclusions valent pour ce modèle, pas pour une antenne réelle construite.
2. **Domaine de validité** : le modèle n'est fiable que pour des spécifications situées dans la plage couverte par les données d'apprentissage.
3. **Ambiguïté du problème inverse** : son étendue n'est pas connue à l'avance ; c'est l'hypothèse qui motive l'approche générative.
4. **Sélection par la physique** : le meilleur-de-K coûte K évaluations du facteur de réseau à l'utilisation.
5. **Difficultés propres aux GAN** : instabilité de l'apprentissage et risque d'effondrement de modes (le générateur ne produit plus qu'une seule configuration). Les stabilisations de la section 5.4 et le critère de validation physique (section 7.4) les atténuent sans les supprimer.
6. **Sorties discrètes** : G produit des réels ; l'arrondi final peut introduire une petite erreur par rapport à la configuration continue qu'il visait.

---

## 11. Synthèse : le modèle en une page

```
                    ┌───────────────────────────── APPRENTISSAGE ─────────────────────────────┐
  Physique Φ  ──▶   paires (c, x), 50 000          Normalisation, découpage par groupes
                                  │
                                  ▼
        z ~ N(0,I) ─┐                                   x réel (dé-quantifié) ─┐
                    ├─▶ G(z, c) ─▶ x̃ ───────────────────────────────────────┬─┤
        c ──────────┘                                                        ▼ ▼
                                                              D( · , c) ─▶ réel / faux
        L_D = BCE(D(x,c),0,9) + ½[BCE(D(x̃,c),0) + BCE(D(x,c̃),0)]      L_G = BCE(D(x̃,c),1)
                    └─────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────── UTILISATION ───────────────────────────────┐
   c souhaité ─▶ normalisation ─▶ K tirages de z ─▶ G ─▶ arrondi / contrainte ─▶ Physique Φ
                                                                                     │
                                         erreur globale E_k  ◀── comparaison avec c ◀┘
                                                     │
                                          x_{k*} = argmin E_k  ─▶  architecture + diagramme
                    └─────────────────────────────────────────────────────────────────────────┘
```

**En une phrase.** Le cGAN apprend, à partir d'exemples fabriqués avec la physique, à *proposer* plusieurs architectures plausibles pour une spécification donnée ; la physique *calcule* ensuite le diagramme réel de chaque proposition et *retient* la meilleure.

---

## Références

- I. Goodfellow et al., « Generative Adversarial Nets », NeurIPS, 2014.
- M. Mirza, S. Osindero, « Conditional Generative Adversarial Nets », arXiv:1411.1784, 2014.
- S. Reed et al., « Generative Adversarial Text to Image Synthesis », ICML, 2016 (discriminateur sensible à la cohérence).
- D. Kingma, J. Ba, « Adam: A Method for Stochastic Optimization », ICLR, 2015.
- C. Balanis, *Antenna Theory: Analysis and Design* (réseau circulaire, facteur de réseau, équations 6-112 à 6-116).
