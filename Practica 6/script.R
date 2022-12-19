# instalar y cargar paquetes
install.packages("readr")
library(readr)


# establecer directorio de trabajo (donde tienes los CSV)
#setwd("C:/Users/alex1/Documents/UNIVERSIDAD/AMD/practicas/p6")


# ----------DATOS TM_Santiago----------

# leer CSV
TM_Santiago <- read_delim("./TM_Santiago.csv", delim = "|", escape_double = FALSE, trim_ws = TRUE)

# ver histograma
hist(TM_Santiago$TM)

summary(TM_Santiago$TM)

# guardamos los cuartiles y calculamos el rango intercuartílico
Q1 <- 13.1
Q3 <- 22.6
IQR <- Q3 - Q1

# calculamos umbrales leves
umbral_leve_arriba <- Q3 + 1.5 * IQR
umbral_leve_abajo <- Q1 - 1.5 * IQR

# calculamos umbrales extremos
umbral_ext_arriba <- Q3 + 3 * IQR
umbral_ext_abajo <- Q1 - 3 * IQR

# eliminamos los valores atípicos extremos

TM_Santiago_limpio <- TM_Santiago[TM_Santiago$TM > umbral_ext_abajo &
                                  TM_Santiago$TM < umbral_ext_arriba, c ("Fecha", "TM")]

# vemos histograma del conjunto limpio
hist(TM_Santiago_limpio$TM)
summary(TM_Santiago_limpio$TM)

# guardamos el conjunto limpio en un CSV 
write.csv(TM_Santiago_limpio, "./TM_Santiago_limpio.csv", row.names=FALSE)


# ----------DATOS P_Santiago----------

# leer CSV
P_Santiago <- read_delim("./P_Santiago.csv", delim = "|", escape_double = FALSE, trim_ws = TRUE)

# ajustamos el formato del campo fecha
P_Santiago$Fecha <- as.Date(P_Santiago$Fecha, format="%d-%m-%Y")

# ver histograma
hist(P_Santiago$P)

summary(P_Santiago$P)


# guardamos los cuartiles y calculamos el rango intercuartílico
P_Q1 <- 984.2
P_Q3 <- 992.3
P_IQR <- P_Q3 - P_Q1

# calculamos umbrales leves
P_umbral_leve_arriba <- P_Q3 + 1.5 * P_IQR
P_umbral_leve_abajo <- P_Q1 - 1.5 * P_IQR

#calculamos umbrales extremos
P_umbral_ext_arriba <- P_Q3 + 3 * P_IQR
P_umbral_ext_abajo <- P_Q1 - 3 * P_IQR

# eliminamos los valores atípicos extremos

P_Santiago_limpio <- P_Santiago[P_Santiago$P > P_umbral_ext_abajo &
                                    P_Santiago$P < P_umbral_ext_arriba, c ("Fecha", "P")]

# vemos histograma del conjunto limpio
hist(P_Santiago_limpio$P)
summary(P_Santiago_limpio$P)

# guardamos el conjunto limpio en un CSV 
write.csv(P_Santiago_limpio, "./P_Santiago_limpio.csv", row.names=FALSE)


# ----------DATOS IRRA_Santiago----------

# leer CSV
IRRA_Santiago <- read_delim("./IRRA_Santiago.csv", delim = "|", escape_double = FALSE, trim_ws = TRUE)

# ajustamos el formato del campo fecha
IRRA_Santiago$Instante_lectura <- as.Date(IRRA_Santiago$Instante_lectura, format="%d-%m-%Y")

# ver histograma
hist(IRRA_Santiago$Irradiacion_global_diaria)

summary(IRRA_Santiago$Irradiacion_global_diaria)


# guardamos los cuartiles y calculamos el rango intercuartílico
IRRA_Q1 <- 984.2
IRRA_Q3 <- 992.3
IRRA_IQR <- IRRA_Q3 - IRRA_Q1

# calculamos umbrales leves
IRRA_umbral_leve_arriba <- IRRA_Q3 + 1.5 * IRRA_IQR
IRRA_umbral_leve_abajo <- IRRA_Q1 - 1.5 * IRRA_IQR

#calculamos umbrales extremos
IRRA_umbral_ext_arriba <- IRRA_Q3 + 3 * IRRA_IQR
IRRA_umbral_ext_abajo <- IRRA_Q1 - 3 * IRRA_IQR

# eliminamos los valores atípicos extremos

IRRA_Santiago_limpio <- IRRA_Santiago[IRRA_Santiago$Irradiacion_global_diaria > IRRA_umbral_ext_abajo &
                                      IRRA_Santiago$Irradiacion_global_diaria < IRRA_umbral_ext_arriba, c("Instante_lectura", "Irradiacion_global_diaria")]

# vemos histograma del conjunto limpio
hist(IRRA_Santiago_limpio$Irradiacion_global_diaria)
summary(IRRA_Santiago_limpio$Irradiacion_global_diaria)

# guardamos el conjunto limpio en un CSV 
write.csv(IRRA_Santiago_limpio, "./IRRA_Santiago_limpio.csv", row.names=FALSE)
