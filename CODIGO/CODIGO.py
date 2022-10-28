"""
El siguiente código es para clasificar una nube de puntos 3D en familias de planos de discontinuidades,
el archivo de entrada debe tener un formato txt. Enseguida se describirán las columnas de un archivo 
de entrada (punto inicial).

x = coordenada x de la nube de puntos
y = coordenada y de la nube de puntos
z = coordenada z de la nube de puntos
R = código de color rojo
G = código de color verde
B = código de color azul
nx = coordenada x del vector normal unitario
ny = coordenada y del vector normal unitario
nz = coordenada z del vector normal unitario
Aquí ejemplificaremos el código con el uso de una nube de puntos que llamamos ciclovía
"""

#Importando las librerías 
#Librerías para el manejo de la información
import numpy as np
import pandas as pd
#Librería para graficar
import matplotlib.pyplot as plt
#Librerías de algoritmos de machine learning
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#Librería para hacer operaciones 
import math 
import scipy
from itertools import combinations
#Librerías calcular distancia
from scipy import spatial
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
#Librería para el manejo de nube de puntos
import open3d as o3d

#Números de decimales con lo0s que realizará las operaciones
pd.options.display.float_format = '{:.8f}'.format

#Leyendo la nube de puntos de un archivo de texto
ciclovia = pd.read_csv('ciclovia.txt',header = None, sep = ' ')

#Se utiliza el criterio del codo para visualizar el número de cluster óptimo (parámetro del K-means).
#Parámetros de la función: (nube = archivo de texto que tiene la nube de puntos 3D).
#Salida: (dif = regresa los cambios de pendientes que se observan en la gráfica, ya que el 
#clúster óptimo es aquel punto donde cambia bruscamente la pendiente de la gráfica).
def elbow(nube):
    wcss = []
    distortions = []
    dif = []
    #Extraemos las coordenadas de los vectores normales de la nube
    nube = nube.loc[:, 6:]
    X = range(1,15)
    for i in X:
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000)
        kmeans.fit(nube)
        wcss.append(kmeans.inertia_)
        
    c = max(wcss)
    for i in range(len(wcss)-1): 
        difer = math.degrees(math.atan(wcss[i]/c - wcss[i+1]/c))
        dif.append(difer) 
          
    plt.plot(X, wcss, 'bx-')
    plt.title(" Elbow Method")
    plt.xlabel("Number of cluster")
    plt.ylabel("Inertia")
    plt.show()
    
    return wcss

#Ejecutando la función elbow
elbow(ciclovia)

#La función agrupamientos asigna a cada punto de la nube en una familia utilizando el algoritmo K-means.
#Parámetros de la función: (nube = nube de puntos, numK = parámetro seleccionado del criterio del codo)
#Salida: (nube_clasificada = un arreglo con los datos etiquetados según la familia a la que pertenecen).

def agrupamientos(nube, numK):
    
    nubecoord = nube.loc[:,:2] #seleccionar las coordenadas de los puntos de la nube
    nubenormales = nube.loc[:, 6:] #seleccionar las coordenadas de los vectores normales
    #Inicializamos el algoritmo Kmeans con los parámetros
    clustering = KMeans(n_clusters = numK, n_init = 100, tol=0.00001)
    clustering.fit(nubenormales)
    #Se extrae en un arreglo que contiene la etiqueta del clúster al que pertenece cada punto
    label = np.array(clustering.labels_)
    #Se crea un arreglo concatenando las coordenadas de los puntos y su etiqueta
    nube_clasificada = np.append(nubecoord, label.reshape(-1, 1), axis=1)
    #Guardamos en un archivo de texto los datos anteriores
    np.savetxt('nube_Clasificada.txt', nube_clasificada) 
    return nube_clasificada


#Ejecutando la función agrupamientos
nube_clasificada = agrupamientos(ciclovia, 3)

#Convertir la nube_clasificada a un DataFrame
nube_clasificada = pd.DataFrame(nube_clasificada)

#La siguiente función es para definir o limpiar los clusters, para ello utilizaremos el 
#algoritmo KNN (k-Nearest Neighbors).
#Parámetros de la función: (nube_clasificada1 = nube clasificada)
#Salida: (nueva nube clasificada)

def definir_cluster(nube_clasificada, k_vecinos):

    X = nube_clasificada.loc[:,:2]
    Y = nube_clasificada[3]
    #Inicialización del algoritmo KNN 
    regressor = KNeighborsClassifier(n_neighbors = k_vecinos)
    regressor.fit(X, Y)
    #Reasignación de etiquetas a cada uno de los puntos según sus k_vecinos más cercanos
    y_pred = regressor.predict(X)
    #Se extrae en un arreglo la etiqueta de asignación de cada punto en un grupo
    cluster = np.array(y_pred)
    #Reemplazar la nueva asignación
    nube_clasificada[3] = cluster
    #Guardar los datos anteriores en un archivo txt
    np.savetxt('cluster'+str(k_vecinos)+'.txt', nube_clasificada)
    return nube_clasificada

#Ejecutando la función definir_cluster 
nube_reclasificada = definir_cluster(nube_clasificada, 11)

#Verificar la nueva nube, si es mejor, se elige para los siguientes pasos.
#Utilizamos el algoritmo DBSCAN para segmentar espacialmente los puntos que pertenecen a una 
#misma familia pero que corresponden a planos diferentes.
#Parámetros del DBSCAN(eps, min_points)
#eps = especifica lo cerca que deben estar los puntos entre sí para ser considerados parte #de un clúster.
#min_points = el número mínimo de puntos para formar una región densa.
#Elegir el valor óptimo del DBSCAN
#De acuerdo con la siguiente gráfica el valor óptimo para eps debe ser aquel donde se #observe un cambio brusco en la gráfica.
#Elegir el mejor valor del DBSCAN EPS 
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(nube_clasificada)
distances, indices = neighbors_fit.kneighbors(nube_clasificada)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

#Parámetros de la función cluster(nube_clasificada = nube que tiene las etiquetas de las 
#familias a las que pertenece cada punto, kmean = valor del criterio del codo, 
#value_eps = valor que se obtuvo en la gráfica anterior, puntos_conjunto = el número mínimo de puntos 
#para que se acepte como conjunto).
#Salida: (familias = archivo de texto por cada plano que se encontró, así como un archivo 
#que contiene toda la información en general).
#Agrupacion por sub_planos
def cluster(nube_clasificada, kmean, value_eps, puntos_conjunto):
    familias = pd.DataFrame()
    
    for i in range(kmean):
        data = pd.DataFrame()
        familia = nube_clasificada.loc[nube_clasificada[3] == i]
        familia_cluster = np.asarray(familia.loc[:,:2])
        #Convertir la nube_clasificada a un objeto de open3d
        nube = o3d.geometry.PointCloud()
        nube.points = o3d.utility.Vector3dVector(familia_cluster)
       #Solamente consideramos familias con más de 1000 puntos
        if len(familia) > 1000:
            labels = np.array(nube.cluster_dbscan(eps= value_eps, min_points = 10))
            unique, counts = np.unique(labels, return_counts=True)
            familia['cluster'] = labels
            for j in range(len(unique)):
            #Solamente consideramos planos formados con más de puntos_conjunto
                if j != 0 and counts[j] > puntos_conjunto:
                    data = pd.DataFrame()
                    fam = familia.loc[familia.cluster == (j-1)]
                    #Asignamos un color a cada plano
                    color = list(np.random.choice(range(256), size=3))
                    data = data.append(fam)
                    data[['R', 'G', 'B']] = color
                    nombre = 'Familia'+str(i)+str(j-1)+'.txt'
                    familias = familias.append(data)
                    np.savetxt(nombre, data)
                 
            np.savetxt('Familias.txt', familias)

    return familias

#Ejecutando la función cluster
familias = cluster(nube_clasificada, 3, 0.02, 1000)

#Leer el archivo que se generó en la función cluster
Familias = pd.read_csv('Familias.txt',header = None, sep = ' ')

#La función dip_direc2 obtiene el dip direction y el dip de los planos encontrados
#Parámetros: (a,b,c = son los respectivos coeficientes de la ecuación de cada plano)
#Salida: (dip_dir, dip = dip direction y el dip de cada plano respectivamente)
def dip_direc2(a, b, c):
    
    if a > 0 and b > 0:
        dip_dir = math.degrees(math.atan(a/b))
    elif a > 0 and b < 0:
        dip_dir = 360 + math.degrees(math.atan(a/b))
    elif a <= 0 and b <= 0:
       dip_dir = 180 + math.degrees(math.atan(a/b))
    elif a < 0 and b > 0:
        dip_dir = 360 - math.degrees(math.acos(abs(b)/math.sqrt((a)**2 + (b)**2)))
         
    dip = math.degrees(math.acos(abs(c)))
    
    if a == 0 and b == 0:
        dip_dir = 0
        dip = 0

    return dip_dir, dip


#Con el algoritmo RANSAC se obtiene la ecuación del plano que se ajusta mejor a los 
#puntos de cada cluster que se encontró ejecutando la función cluster.
#Parámetros del RANSAC (distance_threshold, ransac_n, num_iterations)
#distance_threshold: es una distancia umbral desde el plano a cada punto para considerarlo como punto
#perteneciente o no al respectivo plano.
#ransac_n = número de puntos muestreados para estimar cada candidato al plano (en nuestro caso 
#siempre será igual a 3).
#num_iterations = número de veces que se seleccionará 3 puntos diferentes para estimar al mejor 
#plano candidato.
#Parámetros de la función planos: (arc_gen = al archivo familias que se obtuvo al ejecutar la función 
#cluster, kmean = valor del criterio del codo)
#Salida: (Planos = un DataFrame que contiene la información de cada plano, familia, subfamilia, 
#coeficientes de la ecuación del plano, dip direction, dip, puntos que lograron ajustarse al plano, 
#traza del plano)
def planos(arc_gen, kmean, dist_thr_va):

    data_planos = pd.DataFrame()
    data_distancia = []
    nubeout = pd.DataFrame()
    #for i in range(kmean):
    for i in arc_gen[3].unique():
        familia = arc_gen.loc[arc_gen[3] == i]
        for j in familia[4].unique():
            sub_familia = familia.loc[(familia[3] == i) & (familia[4] == j)]
            cluster = np.asarray(sub_familia.loc[:,:2])
            nube = o3d.geometry.PointCloud()
            nube.points = o3d.utility.Vector3dVector(cluster)
            plano, inliers = nube.segment_plane(distance_threshold = dist_thr_va, ransac_n=3, num_iterations=10000)
            [a,b,c,d] = plano
            puntos_inliers = nube.select_by_index(inliers)
            sub_nube = np.asarray(puntos_inliers.points)
            nuboutt = pd.DataFrame(data=sub_nube, index=None, columns=None)
            nuboutt[['familia', 'sub_familia']] = [i,j]
            nubeout = nubeout.append(nuboutt, ignore_index=True)
            
            x = nuboutt.loc[:, :2].to_numpy()
            traza = pdist(x, 'euclidean').max()
                
            dip_dir, dip = dip_direc2(a, b, c)
            lista_datos = pd.Series([i,j,a,b,c,d,dip_dir, dip, nuboutt.shape[0], traza])
            data_planos = pd.concat([data_planos, lista_datos], axis = 1)
    #Obtenemos el archivo para poder encontrar la distancia entre planos de la misma familia
    np.savetxt('nubeouts.txt', nubeout)
    #Formato al archivo
    Planos = data_planos.T
    Planos.columns = ['familia', 'sub_planos', 'a', 'b', 'c', 'd','dip_direct','dip','puntos_plano','traza']                                
    Planos.index = list(range(1, Planos.shape[0]+1))  
    Planos.to_csv('Planos.txt')
    return Planos


#Ejecutamos la funcion planos
planos = planos(Familias, 3, 0.1)

#Leer el archivo que tiene los puntos que conforman a cada subplano
sub_planos = pd.read_csv('nubeouts.txt',header = None, sep = ' ')
sub_planos.columns = ['x', 'y','z', 'familia','sub_familia']

planos = pd.read_csv('Planos.txt', sep = ',', index_col=[0])

#Los planos de la misma familia se consideran paralelos. Se calcula la distancia promedio 
#respecto al plano que tiene más puntos de ajuste, con respecto a cada punto que forma otro plano paralelo.
#Parámetros: (a,b,c,d = los coeficientes del mejor plano, puntos_x, puntos_y, puntos_z = las 
#coordenadas de los puntos que pertenecen a la familia f y subplano g)
#Salida: (f,g, promedio = familia a la que pertenece el plano, indicación de la subfamilia y 
#distancia promedio al mejor plano que describe a la familia respectivamente)

def distancia_planos(a,b,c,d, puntos_x, puntos_y, puntos_z, f, g):
    #distancia_p = pd.DataFrame()
    data_distancia = []
    #tabla_distancia = pd.DataFrame()
    for i in range(len(puntos_x)):
        distancia = math.fabs(a*puntos_x[i] + b*puntos_y[i] + c*puntos_z[i] + d)/math.sqrt(a**2 + b**2 + c**2)                                                                                                
        data_distancia.append(distancia)
        lista= np.array(data_distancia)
    
    promedio = lista.mean()
    lista_datos = pd.DataFrame([[f, g, promedio]])
    #distancia_p = pd.concat([distancia_p, lista_datos], axis = 1).T
    #np.savetxt('distancia_planos.txt', distancia_p) #Familia, sub_planos, distancia entre planos
    print(f,g, promedio)
    return lista_datos

#La función dist_planos elige el mejor plano y selecciona los puntos de los planos 
#pertenecientes a los planos de la misma familia.
#Paramétros: (planos = archivo que se obtiene de ejecutar la función planos y sub_planos 
#archivo que tiene información de las coordenadas, familia y subfamilia a la que pertenece #cada punto       
def dist_planos(planos, sub_planos):
    distancia_p = pd.DataFrame()
    for i in planos['familia'].unique():
        familia_planos = planos.loc[planos['familia'] == i]
        plano_principal = familia_planos.loc[familia_planos['puntos_plano'] == familia_planos['puntos_plano'].max()].reset_index()                          
        a = plano_principal.at[0,'a']
        b = plano_principal.at[0,'b']
        c = plano_principal.at[0,'c']
        d = plano_principal.at[0,'d']
        f = plano_principal.at[0,'familia']
        for j in familia_planos['sub_planos'].unique():
            if j != plano_principal.at[0, 'sub_planos']:
                puntos = pd.DataFrame(sub_planos.loc[(sub_planos.familia == i) & (sub_planos.sub_familia == j)])
                puntos_x = puntos['x'].to_list()
                puntos_y = puntos['y'].to_list()
                puntos_z = puntos['z'].to_list()
                lista_datos = distancia_planos(a,b,c,d, puntos_x, puntos_y, puntos_z,f,j)
            else:
                pass
        
        distancia_p = distancia_p.append(lista_datos, ignore_index=True)
    np.savetxt('distancia_planos.txt', distancia_p)

#Ejecución de la función dist_subplanos
dist_planos(planos,sub_planos)

