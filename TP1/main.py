import numpy as np
import sklearn
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2


#mahaut.gerard@eurecom.fr
#ixer le seed
np.random.seed(42)

#Créer une array numpy, X, de 1000 points avec valeur aléatoire dans  l’intervalle [0, 3]
X = np.random.uniform(0, 3, 1000)

# Calculer la moyenne
moyenne = np.mean(X)

# Calculer l'écart type
ecartType = np.std(X)

# Calculer la médiane
mediane = np.median(X)

# Arrondir les valeurs au centième
moyenne = round(moyenne, 2)
ecartType = round(ecartType, 2)
mediane = round(mediane, 2)

# Afficher les résultats
print(f"Moyenne : {moyenne}")
print(f"Écart type : {ecartType}")
print(f"Médiane : {mediane}")

#Créer une array numpy, X_bis, de 1000 points avec valeur aléatoire dans  l’intervalle [0, 3]
X_bis = np.random.uniform(0, 3, 1000)

# Calculer la moyenne
moyenne_bis = np.mean(X_bis)

# Calculer l'écart type
ecartType_bis = np.std(X_bis)

# Calculer la médiane
mediane_bis = np.median(X_bis)

# Arrondir les valeurs au centième
moyenne_bis = round(moyenne_bis, 2)
ecartType_bis = round(ecartType_bis, 2)
mediane_bis = round(mediane_bis, 2)

# Afficher les résultats
print(f"Moyenne : {moyenne_bis}")
print(f"Écart type : {ecartType_bis}")
print(f"Médiane : {mediane_bis}")


min_moy = min(moyenne, moyenne_bis)
max_moy = max(moyenne, moyenne_bis)

min_et = min(ecartType, ecartType_bis)
max_et = max(ecartType, ecartType_bis)

min_m = min(mediane, mediane_bis)
max_m = max(mediane, mediane_bis)

# Calculer la liste y = sin(X) + bruit gaussien
bruit_gaussien = np.random.normal(0, 0.1, 1000)
y = np.sin(X) + bruit_gaussien


# Visualiser y en fonction de X sous forme de graph ‘scatter’
plt.figure(figsize=(10, 6))  # Changer la taille de la figure
plt.scatter(X, y, alpha=0.5)
plt.title('Scatter Plot de y en fonction de X')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Visualiser le bruit assign sous forme d’histogramme
plt.figure(figsize=(10, 6))  # Changer la taille de la figure
plt.hist(bruit_gaussien, bins=50, edgecolor='k')
plt.title('Histogramme du Bruit Gaussien')
plt.xlabel('Valeur')
plt.ylabel('Fréquence')
plt.show()


# Lecture de l'image
img = cv2.imread('image_1.jpg')

# Affichage de l'image en couleur
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertion de l'image en noir et blanc
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Afficher l'image en noir et blanc
cv2.imshow('Image en Noir et Blanc', img_bw )
cv2.waitKey(0)  # Attendre une touche
cv2.destroyAllWindows()  # Fermer la fenêtre


#SEUILLAGE
# Définir les seuils minimum et maximum
seuil_min = 200  # Par exemple
seuil_max = 255  # Valeur maximale pour garder les blancs


# Appliquer le seuillage
_, img_seuillage = cv2.threshold(img_bw, seuil_min, seuil_max, cv2.THRESH_BINARY)

# Enregistrer l'image en noir et blanc
cv2.imwrite('image_seuillage.jpg', img_seuillage)

# Afficher l'image en noir et blanc
cv2.imshow('Image Seuillée', img_seuillage)
cv2.waitKey(0)
cv2.destroyAllWindows()



#FILTRE DE SOBEL
# Appliquer le filtre de Sobel pour détecter les contours horizontaux et verticaux
sobel_x = cv2.Sobel(img_bw, cv2.CV_64F, 1, 0, ksize=5)  # Gradient horizontal
sobel_y = cv2.Sobel(img_bw, cv2.CV_64F, 0, 1, ksize=5)  # Gradient vertical

# Calculer la magnitude du gradient
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

# Afficher les résultats
cv2.imshow('Sobel Magnitude', sobel_mag)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Filtre DE CANNY
edges_canny = cv2.Canny(img_bw, 100, 200)

# Afficher les résultats
cv2.imshow('Sobel Magnitude', sobel_mag)
cv2.imshow('Canny Edges', edges_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()